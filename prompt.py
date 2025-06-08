"""Prompt for extracting a knowledge graph from text.

Running this module will print the prompt text.
"""

PROMPT = """
You are a helpful assistant that extracts structured knowledge from raw
document text.

Given a DOCUMENT, perform these steps:
1. Identify all entity\-relation\-entity triplets mentioned in the text.
2. For each unique entity, determine a TOPIC and SUBTOPIC that best describe
   the entity based on the surrounding context in the document.
3. For each triplet, list the sentence or sentences from the document that
   mention both entities.

Respond in JSON with two top\-level keys:
- "triplets": a list of objects each containing "entity1", "relation",
  "entity2", and a list of "sentences".
- "entities": a mapping of entity name to an object with "topic" and
  "subtopic".

Use the following format exactly:
{
  "triplets": [
    {
      "entity1": "...",
      "relation": "...",
      "entity2": "...",
      "sentences": ["..."]
    }
  ],
  "entities": {
    "EntityName": {
      "topic": "...",
      "subtopic": "..."
    }
  }
}

DOCUMENT:\n"""

if __name__ == "__main__":
    print(PROMPT)
