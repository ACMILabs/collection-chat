# ACMI collection chat

Uses LangChain and GPT-4 to chat with the ACMI Public API collection.

## Run it on your computer

* Build the virtual environment: `make build`
* Setup your OpenAI API Key: `cp template.envrc .envrc`
* Load the environment `direnv allow`
* Start chatting: `make up`

Or if you have your own Python environment setup:

* Install the dependencies: `pip install -r requirements.txt`
* Start chatting: `python chat.py`

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
