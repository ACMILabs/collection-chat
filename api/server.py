#!/usr/bin/env python
"""ACMI Public API LangChain server exposes a conversational retrieval chain.

References:
* https://python.langchain.com/docs/expression_language/cookbook/retrieval
* https://github.com/langchain-ai/langserve/blob/main/examples/chat_playground/server.py

To run this example:
make build
make server
"""

import json as json_parser
import os
import random as random_module
from operator import itemgetter
from typing import List, Tuple

from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

DATABASE_PATH = os.getenv('DATABASE_PATH', '')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'works')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', 'works_db')
MODEL = os.getenv('MODEL', 'gpt-4o')
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', None)
LLM_BASE_URL = os.getenv('LLM_BASE_URL', None)
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'
DOCUMENT_IDS = []
NUMBER_OF_RESULTS = int(os.getenv('NUMBER_OF_RESULTS', '6'))
CHAT_PORT = int(os.getenv('CHAT_PORT', '8000'))

_TEMPLATE = """Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. This might mean summarising the context in the
chat history within the standalone question. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Please use your general knowledge if the question includes the title of a film,
tv show, videogame, artwork, or object.

If you find relevant information please use the ID (not the ACMI ID) of the collection item to
create a link for the title of the item.
e.g. <a href="https://url.acmi.net.au/w/<ID>"><Title></a>

Please take on the personality of ACMI museum CEO Seb Chan and reply in a form suitable
to be spoken by a text-to-speech engine.

Please output the response in valid raw HTML, but don't wrap the response in
Markdown, just the raw HTML.

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template='{page_content}')


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator='\n\n'
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = 'Human: ' + dialogue_turn[0]
        assistant = 'Assistant: ' + dialogue_turn[1]
        buffer += '\n' + '\n'.join([human, assistant])
    return buffer


def get_random_documents(
        document_search,
        number_of_documents: int = NUMBER_OF_RESULTS,
) -> List[any]:
    """Fetch random documents from the vector store."""
    random_ids = random_module.sample(DOCUMENT_IDS, min(number_of_documents, len(DOCUMENT_IDS)))
    random_documents = document_search.get(random_ids)['documents']
    return random_documents


embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL or 'text-embedding-ada-002')

if MODEL.startswith('gpt'):
    llm = ChatOpenAI(temperature=0, model=MODEL)
elif MODEL.startswith('o'):
    llm = ChatOpenAI(temperature=1, model=MODEL)
else:
    llm = Ollama(model=MODEL)
    if EMBEDDINGS_MODEL:
        embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    if LLM_BASE_URL:
        llm.base_url = LLM_BASE_URL
        embeddings.base_url = LLM_BASE_URL

docsearch = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=f'{DATABASE_PATH}{PERSIST_DIRECTORY}',
)
retriever = docsearch.as_retriever(search_kwargs={'k': NUMBER_OF_RESULTS})
DOCUMENT_IDS = docsearch.get()['ids']

_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x['chat_history'])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)
_context = {
    'context': itemgetter('standalone_question') | retriever | _combine_documents,
    'question': lambda x: x['standalone_question'],
}


# User input
class ChatHistory(BaseModel):  # pylint: disable=too-few-public-methods
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={'widget': {'type': 'chat', 'input': 'question'}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title='ACMI Public API LangChain Server',
    version='1.0',
    description='A simple ACMI Public API chat server using Langchain\'s Runnable interfaces.',
)

# Load static assets and templates
app.mount('/static', StaticFiles(directory='api/static'), name='static')
templates = Jinja2Templates(directory='api/templates')

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['*'],
)

# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=False)


@app.get('/')
async def root(
    request: Request,
    json: bool = True,
    query: str = '',
    random: bool = False,
):
    """Returns the home view."""

    results = []
    home_json = {
        'message': 'Welcome to the ACMI Collection Chat API.',
        'api': sorted({route.path for route in app.routes}),
        'acknowledgement':
            'ACMI would like to acknowledge the Traditional Custodians of the lands '
            'and waterways of greater Melbourne, the people of the Kulin Nation, and '
            'recognise that ACMI is located on the lands of the Wurundjeri people. '
            'First Nations (Aboriginal and Torres Strait Islander) people should be aware '
            'that this website may contain images, voices, or names of deceased persons in '
            'photographs, film, audio recordings or text.',
    }

    if query:
        results = [json_parser.loads(result.page_content) for result in retriever.invoke(query)]

    if random:
        results = [json_parser.loads(result) for result in get_random_documents(docsearch)]

    if json and query:
        return results

    if json:
        return home_json

    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={'query': query, 'results': results, 'model': MODEL},
    )


@app.post('/similar')
async def similar(request: Request):
    """Returns similar items from the vector database to the body string."""

    body = await request.body()
    body = body.decode('utf-8')
    responses = docsearch.similarity_search_with_relevance_scores(body, k=NUMBER_OF_RESULTS)
    results = []
    for response in responses:
        result = json_parser.loads(response[0].page_content)
        result['score'] = response[1]
        results.append(result)
    return results


@app.post('/speak')
async def speak(request: Request):
    """Returns an audio stream of the body string."""

    text_to_speech = ElevenLabs()
    body = await request.body()
    body = body.decode('utf-8')
    return StreamingResponse(text_to_speech.generate(
        text=body,
        voice='Seb Chan',
        model='eleven_flash_v2_5',
        stream=True,
    ), media_type="audio/mpeg")


@app.post('/summarise')
async def summarise(request: Request):
    """Returns a summary of the visitor's query vs the results, including an anecdote."""
    body = await request.body()
    body = body.decode('utf-8')
    llm_prompt = f"""
        System: You are an ACMI museum guide. Please compare the visitor's question to the museum
        collection items in the response and provide a brief summary as if you were talking
        to the visitor in a short one sentence form suitable for text-to-speech as it will be
        converted to audio and read back to the visitor.

        Provide an anecdote about the data in one or more of the collection records.

        Example: Did you know <summary>. <anecdote>

        User's query and context:

        {body}
    """
    return llm.invoke(llm_prompt).content.replace('\"', '"')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=CHAT_PORT)
