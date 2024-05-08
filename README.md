# ACMI collection chat

![ACMI Collection Chat CI](https://github.com/ACMILabs/collection-chat/workflows/ACMI%20Collection%20Chat%20CI/badge.svg)

Uses LangChain, LangServe, and GPT-4 to chat with the ACMI Public API collection.

## Run it on your computer

* Build the virtual environment: `make build-local`
* Setup your OpenAI API Key: `cp config.tmpl.env config.env`
* Install direnv if you'd like to load API Keys from the `config.env` file: `brew install direnv`
* Load the environment `direnv allow`
* Start chatting on the command line: `make up-local`
* Start chatting in a web browser: `make server-local` and visit: http://localhost:8000/playground
* See the API server documentation: http://localhost:8000/docs

Or if you have your own Python environment setup:

* Install the dependencies: `pip install -r requirements.txt`
* Start chatting on the command line: `python chat.py`
* Start chatting in a web browser: `python api/server.py` and visit: http://localhost:8000/playground
* See the API server documentation: http://localhost:8000/docs

Or to run the API server using Docker:

* Build: `make build`
* Run: `make up`
* Visit: http://localhost:8000
* Clean-up: `make down`

### Environment variables

Optional environment variables you can set:

* `DATABASE_PATH` - set where your Chromadb vector database is located
* `COLLECTION_NAME` - the name of the Chromadb collection to save your data to
* `PERSIST_DIRECTORY` - the name of the directory to save your persistant Chromadb data
* `MODEL` - the OpenAI chat model to use
* `REBUILD` - set to `true` to rebuild your Chromadb vector database
* `ALL` - set to `true` to rebuild with the entire ACMI Public API collection

When using Docker:

* `CACHE_URL` - the URL to your pre-generated embeddings database e.g. https://example.com/path/

The `scripts/entrypoint.sh` script will look for a file named `works_db_chat.tar.gz` at the `CACHE_URL` to process.

To create that file, generate your embeddings database locally and then run: `tar -cvzf works_db_chat.tar.gz works_db`

### Open source LLM using Ollama

To use an open source model on your computer with [Ollama](https://ollama.com):

* Install `ollama` with: `brew install ollama`
* Pull an open source model: `ollama pull llama3`
* Start the `ollama` server with: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`
* Set `MODEL=llama3`
* Set `LLM_BASE_URL=http://<YOUR_IP_ADDRESS>:11434`
* Start the chat: `make up`

### Re-build all collection items

By default we only build the first ten pages of the Public API into the vector database, but if you'd like the build the entire collection:

* Add `ALL=true` to your `config.env`
* Then either delete your persistant directory or also add `REBUILD=true` to your `config.env`
* Rebuild the app: `make build-local`

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
