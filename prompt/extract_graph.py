EXTRACTION_PROMPT = """
You are a highly skilled information extraction system.

---Goal---
Extract factual (subject, relation, object) triples from the document and classify the **subject** and **object** into a subtopic and a main topic.

---Instructions---
1. Read the document below and extract all factual (subject, relation, object) triples. Each triple must be grounded in a specific sentence from the document.
2. Paraphrasing is acceptable **only** if the relation is implied by the sentence.
3. Resolve pronouns such as "it", "he", "she", "they", etc. using surrounding context, and use the resolved entity names in the triples.
4. For each subject and object:
   - Assign a **Subtopic** (a specific category)
   - Assign a **Main topic** (a broader domain)
   - Subtopic and main topic should be relevant to the subject and object, respectively.
   - Also, subtopic and main topic should be relevant to the document as a whole.
5. Return only valid JSON in the specified format. Do not include markdown, comments, or any other text.
6. Do not include any additional text or explanations outside the JSON format.
7. Ensure that the JSON is well-formed and valid.

---Output Format---
Respond ONLY with a valid JSON array using the following format:

[
  {
    "triple": ["Subject", "Relation", "Object"],
    "sentence": "Exact sentence from which the triple was extracted.",
    "subject": {
      "subtopic": "Subtopic Name",
      "main_topic": "Main Topic Name"
    },
    "object": {
      "subtopic": "Subtopic Name",
      "main_topic": "Main Topic Name"
    }
  },
  ...
]

---Example---
Document:
"Marie Curie discovered radium in 1898. Her research contributed significantly to the field of radioactivity."

Output:
[
  {
    "triple": ["Marie Curie", "discovered", "radium"],
    "sentence": "Marie Curie discovered radium in 1898.",
    "subject": {
      "subtopic": "Scientist",
      "main_topic": "Science History"
    },
    "object": {
      "subtopic": "Chemical Element",
      "main_topic": "Science"
    }
  },
  {
    "triple": ["Her research", "contributed to", "the field of radioactivity"],
    "sentence": "Her research contributed significantly to the field of radioactivity.",
    "subject": {
      "subtopic": "Scientific Work",
      "main_topic": "Scientific Research"
    },
    "object": {
      "subtopic": "Physics Subfield",
      "main_topic": "Science"
    }
  }
]

---Input Document---
{{document}}
"""
