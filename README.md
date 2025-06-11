# KGRAG
## prompt.py

`prompt.py` contains a text prompt for a language model. The prompt
instructs the model to:

1. Extract entity--relation--entity triplets from a document.
2. Assign a topic and subtopic for each entity.
3. Link each triplet to the sentences in which the two entities appear.

### Usage

Running the module prints the prompt text so it can be copied into an LLM
request:

```
python3 prompt.py
```

You can also import `PROMPT` from `prompt.py` in your own code.

## graph_based_rag.py

`graph_based_rag.py` provides a simple example of retrieval augmented generation over the `graph_v6.gexf` knowledge graph. Each edge now stores the sentence from which the relation was extracted. The script loads the graph, retrieves the nodes that are most relevant to a question, and optionally uses the OpenAI API to generate an answer from the associated triples. If the `qdrant-client` package is installed, an in-memory Qdrant database accelerates similarity search.

### Example

```
python3 graph_based_rag.py "Who does Storey Publishing serve?"
```

The output shows the retrieved triples along with the sentence where each relation was mentioned.

Set the `OPENAI_API_KEY` environment variable if you want to generate answers with the OpenAI API. Without the key, the script will just show the retrieved context.
Install `qdrant-client` to enable faster vector search:

```
pip install qdrant-client
```
