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


COLLECTION_NAME = 'works'
PERSIST_DIRECTORY = 'works_db'
REBUILD = False

# Set true if you'd like langchain tracing via LangSmith https://smith.langchain.com
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

embeddings = OpenAIEmbeddings()
docsearch = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
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

print('=========================')
print('ACMI collection chat v0.1')
print('=========================\n')

while True:
    try:
        query = input('Question: ')
        response = retrieval_qa.invoke(query)
        try:
            print(f'Answer: {json.loads(response["result"])["answer"]}')
            print(f'Sources: {json.loads(response["result"])["sources"]}\n\n')
        except TypeError:
            print(f'Answer: {response}')
    except KeyboardInterrupt:
        print('\nNice chatting to you.\n')
        break
