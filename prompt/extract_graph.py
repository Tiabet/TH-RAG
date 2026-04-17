EXTRACTION_PROMPT = """
You are an information extraction system.

Goal:
Extract factual (subject, relation, object) triples from the document and assign a subject and object subtopic plus main topic.

Instructions:
1. Read the full document.
2. Extract factual triples grounded in specific sentences.
3. Resolve pronouns so the triples use explicit referents.
4. For the subject and object of each triple, assign:
   - subtopic: a specific category
   - main_topic: a broader category
5. Return only valid JSON.

Expected format:
[
  {
    "triple": ["subject", "relation", "object"],
    "sentence": "Supporting sentence from the document.",
    "subject": {
      "subtopic": "Specific category",
      "main_topic": "Broader category"
    },
    "object": {
      "subtopic": "Specific category",
      "main_topic": "Broader category"
    }
  }
]

Example document:
"Inez is an Estonian new media artist. She lives in Finland. She has composed electronic music."

Example output:
[
  {
    "triple": ["Inez", "is", "an Estonian new media artist"],
    "sentence": "Inez is an Estonian new media artist.",
    "subject": {
      "subtopic": "New Media Artist",
      "main_topic": "Art"
    },
    "object": {
      "subtopic": "Nationality",
      "main_topic": "Culture"
    }
  },
  {
    "triple": ["Inez", "lives in", "Finland"],
    "sentence": "She lives in Finland.",
    "subject": {
      "subtopic": "New Media Artist",
      "main_topic": "Art"
    },
    "object": {
      "subtopic": "Country",
      "main_topic": "Geography"
    }
  }
]

Input document:
{{document}}
"""
