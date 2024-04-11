# -*- coding: utf-8 -*-
"""ACMI collection chat

Uses Langchain RetrievalQA to chat with our collection data.
https://github.com/langchain-ai/langchain/blob/master/cookbook/openai_functions_retrieval_qa.ipynb

Colab prototype: https://colab.research.google.com/drive/1RLe2LliEE63KaQgxXDv3xccmxCYpmmPx
"""

import json
import os
import requests

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory


DATABASE_PATH = os.getenv('DATABASE_PATH', '')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'works')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', 'works_db')
REBUILD = os.getenv('REBUILD', 'false').lower() == 'true'
HISTORY = os.getenv('HISTORY', 'true').lower() == 'true'

# Set true if you'd like langchain tracing via LangSmith https://smith.langchain.com
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

embeddings = OpenAIEmbeddings()
docsearch = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=f'{DATABASE_PATH}{PERSIST_DIRECTORY}',
)

if len(docsearch) < 1 or REBUILD:
    # First 10 pages of works on the ACMI Public API
    PAGES = 10
    json_data = {
        'results': [],
    }
    for index in range(2, (PAGES + 1)):
        page_data = requests.get(
            'https://api.acmi.net.au/works/',
            params={'page': index},
            timeout=10,
        ).json()
        json_data['results'].extend(page_data['results'])
    TMP_FILE_PATH = 'data.json'
    with open(TMP_FILE_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file)
    json_loader = JSONLoader(
        file_path=TMP_FILE_PATH,
        jq_schema='.results[]',
        text_content=False,
    )
    data = json_loader.load()
    for i, item in enumerate(data):
        item.metadata['source'] = f'https://api.acmi.net.au/works/{json_data["results"][i]["id"]}'

    docsearch = Chroma.from_documents(
        data,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

llm = ChatOpenAI(temperature=0, model='gpt-4-turbo-preview')
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
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\
    Make sure to avoid using any unclear pronouns.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
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
