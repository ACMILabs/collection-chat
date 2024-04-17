#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval chain.

Follow the reference here:

https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain

To run this example, you will need to install the following packages:
pip install langchain openai faiss-cpu tiktoken
"""  # noqa: F401

import os
from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field


DATABASE_PATH = os.getenv('DATABASE_PATH', '')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'works')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', 'works_db')

_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Please use your general knowledge if the question includes the title of a film,
tv show, videogame, artwork, or object.

Please include the ID of the collection item in response if you find relevant information.
Also include a link at the bottom with this format: 'https://url.acmi.net.au/w/<ID>'

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
        ai = 'Assistant: ' + dialogue_turn[1]
        buffer += '\n' + '\n'.join([human, ai])
    return buffer


embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model='gpt-4-turbo-2024-04-09')
docsearch = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=f'{DATABASE_PATH}{PERSIST_DIRECTORY}',
)
retriever = docsearch.as_retriever()

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
class ChatHistory(BaseModel):
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

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
