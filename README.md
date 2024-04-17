# ACMI collection chat

Uses LangChain and GPT-4 to chat with the ACMI Public API collection.

## Run it on your computer

* Build the virtual environment: `make build`
* Setup your OpenAI API Key: `cp template.envrc .envrc`
* Install direnv if you'd like to load API Keys from the `.envrc` file: `brew install direnv`
* Load the environment `direnv allow`
* Start chatting on the command line: `make up`
* Start chatting in a web browser: `make server` and visit: http://localhost:8000/playground
* See the API server documentation: http://localhost:8000/docs

Or if you have your own Python environment setup:

* Install the dependencies: `pip install -r requirements.txt`
* Start chatting on the command line: `python chat.py`
* Start chatting in a web browser: `python api/server.py` and visit: http://localhost:8000/playground
* See the API server documentation: http://localhost:8000/docs

### Environment variables

Optional environment variables you can set:

* `DATABASE_PATH` - set where your Chromadb vector database is located
* `COLLECTION_NAME` - the name of the Chromadb collection to save your data to
* `PERSIST_DIRECTORY` - the name of the directory to save your persistant Chromadb data
* `MODEL` - the OpenAI chat model to use
* `REBUILD` - set to `true` to rebuild your Chromadb vector database
* `ALL` - set to `true` to rebuild with the entire ACMI Public API collection

### Re-build all collection items

By default we only build the first ten pages of the Public API into the vector database, but if you'd like the build the entire collection:

* Add `ALL=true` to your `.envrc`
* Then either delete your persistant directory or also add `REBUILD=true` to your `.envrc`
* Rebuild the app: `make build`

## Run it on Google Colab

Chat with the first page of the ACMI Public API: [Google Colab](https://colab.research.google.com/drive/1RLe2LliEE63KaQgxXDv3xccmxCYpmmPx).

## Sample output

```bash
python chat.py
=========================
ACMI collection chat v0.1
=========================

Question: What was the collection item that has gnomes in it?
Answer: The collection item that has gnomes in it is titled "Toy magic lantern slide (Gnomes in umbrellas on water)". It is a work from Germany, circa 1900, and was last on display at ACMI: Gallery 1 on June 23, 2023. The item is categorized under the curatorial section "The Story of the Moving Image → Moving Pictures → MI-02. Play and Illusion → MI-02-C01 → Panel C8" and has measurements of 3.5 x 14.3cm. It is a 2D Object, specifically a Glass slide/Pictorial.
Sources: ['https://api.acmi.net.au/works/119591']
```
