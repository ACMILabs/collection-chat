# -*- coding: utf-8 -*-
"""ACMI collection chat

Uses Langchain RetrievalQA to chat with our collection data.
https://github.com/langchain-ai/langchain/blob/master/cookbook/openai_functions_retrieval_qa.ipynb

Colab prototype: https://colab.research.google.com/drive/1RLe2LliEE63KaQgxXDv3xccmxCYpmmPx
"""

import json
import os

import openai
import requests
from furl import furl
from langchain.chains import (ConversationalRetrievalChain, LLMChain,
                              RetrievalQA, create_qa_with_sources_chain)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DATABASE_PATH = os.getenv('DATABASE_PATH', '')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'works')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', 'works_db')
MODEL = os.getenv('MODEL', 'gpt-4-turbo-2024-04-09')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', None)
REBUILD = os.getenv('REBUILD', 'false').lower() == 'true'
HISTORY = os.getenv('HISTORY', 'true').lower() == 'true'
ALL = os.getenv('ALL', 'false').lower() == 'true'

# Set true if you'd like langchain tracing via LangSmith https://smith.langchain.com
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

if MODEL.startswith('gpt'):
    llm = ChatOpenAI(temperature=0, model=MODEL)
    embeddings = OpenAIEmbeddings()
else:
    llm = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)
    if LLM_BASE_URL:
        llm.base_url = LLM_BASE_URL
        embeddings.base_url = LLM_BASE_URL

docsearch = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=f'{DATABASE_PATH}{PERSIST_DIRECTORY}',
)

if len(docsearch) < 1 or REBUILD:
    json_data = {
        'results': [],
    }
    params = {'page': ''}
    if ALL:
        print('Loading all of the works from the ACMI Public API')
        while True:
            page_data = requests.get(
                'https://api.acmi.net.au/works/',
                params=params,
                timeout=10,
            ).json()
            json_data['results'].extend(page_data['results'])
            if not page_data.get('next'):
                break
            params['page'] = furl(page_data.get('next')).args.get('page')
            if len(json_data['results']) % 1000 == 0:
                print(f'Downloaded {len(json_data["results"])}...')
    else:
        print('Loading the first ten pages of works from the ACMI Public API')
        PAGES = 10
        json_data = {
            'results': [],
        }
        for index in range(1, (PAGES + 1)):
            page_data = requests.get(
                'https://api.acmi.net.au/works/',
                params=params,
                timeout=10,
            )
            json_data['results'].extend(page_data.json()['results'])
            print(f'Downloaded {page_data.request.url}')
            params['page'] = furl(page_data.json().get('next')).args.get('page')
    print(f'Finished downloading {len(json_data["results"])} works.')

    TMP_FILE_PATH = 'data.json'
    with open(TMP_FILE_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file)
    json_loader = JSONLoader(
        file_path=TMP_FILE_PATH,
        jq_schema='.results[]',
        text_content=False,
    )
    data = json_loader.load()

    # Add source metadata
    for i, item in enumerate(data):
        item.metadata['source'] = f'https://api.acmi.net.au/works/{json_data["results"][i]["id"]}'

    def chunks(input_list, number_per_chunk):
        """Yield successive chunks from the input_list."""
        for idx in range(0, len(input_list), number_per_chunk):
            yield input_list[idx:idx + number_per_chunk]

    # Add to the vector database in chunks to avoid OpenAI rate limits
    for i, sublist in enumerate(chunks(data, 10)):
        docsearch.add_documents(
            sublist,
        )
        print(f'Added {len(sublist)} items to the database... total {(i + 1) * len(sublist)}')
    print(f'Finished adding {len(data)} items to the database')

qa_chain = create_qa_with_sources_chain(llm)
doc_prompt = PromptTemplate(
    template='Content: {page_content}\nSource: {source}',
    input_variables=['page_content', 'source'],
)
final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name='context',
    document_prompt=doc_prompt,
)
retrieval_qa = RetrievalQA(
    retriever=docsearch.as_retriever(),
    combine_documents_chain=final_qa_chain,
)

if HISTORY:
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    TEMPLATE = """Given the following conversation and a follow up question, rephrase the \
    follow up question to be a standalone question, in its original language.\
    Make sure to avoid using any unclear pronouns.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(TEMPLATE)
    condense_question_chain = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
    )

    retrieval_qa = ConversationalRetrievalChain(
        question_generator=condense_question_chain,
        retriever=docsearch.as_retriever(),
        memory=memory,
        combine_docs_chain=final_qa_chain,
    )

print('=========================')
print('ACMI collection chat v0.1')
print('=========================\n')

while True:
    try:
        query = input('Question: ')
        if HISTORY:
            response = retrieval_qa.invoke({'question': query}).get('answer')
        else:
            response = retrieval_qa.invoke(query).get('result')
        try:
            print(f'Answer: {json.loads(response)["answer"]}')
            print(f'Sources: {json.loads(response)["sources"]}\n')
        except TypeError:
            print(f'Answer: {response}\n')
    except KeyboardInterrupt:
        print('\n\nNice chatting to you.\n')
        break
    except openai.BadRequestError:
        print('\n\nSorry, something went wrong with the OpenAI request.\n')
        break
