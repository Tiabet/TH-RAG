EXTRACTION_PROMPT = """
You are a highly skilled information extraction system.

---Goal---
Extract factual (subject, relation, object) triples from the document.

---Instructions---
1. Read the document below and extract all factual (subject, relation, object) triples. Each triple must be grounded in a specific sentence from the document.
2. Paraphrasing is acceptable **only** if the relation is implied by the sentence.
3. Return only valid JSON in the specified format. Do not include markdown, comments, or any other text.
4. Ensure that the JSON is well-formed and valid.

---Output Format---
Respond ONLY with a valid JSON array using the following format:

[
  {
    "triple": ["Subject", "Relation", "Object"],
    "sentence": "Exact sentence from which the triple was extracted."
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
    "sentence": "Marie Curie discovered radium in 1898."
  },
  {
    "triple": ["Her research", "contributed to", "the field of radioactivity"],
    "sentence": "Her research contributed significantly to the field of radioactivity."
  }
]

---Input Document---
{{document}}
"""
