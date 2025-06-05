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
