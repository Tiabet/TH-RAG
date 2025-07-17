EXTRACTION_PROMPT = """
You are a highly skilled information extraction system designed to process factual information accurately and clearly.

---Goal---
Extract factual (subject, relation, object) triples from the document and classify the **subject** and **object** into a subtopic and a main topic.

---Instructions---
1. Read the entire document below and extract all factual (subject, relation, object) triples. Each triple must be grounded in a specific sentence from the document.
2. Paraphrasing is acceptable **only** if the relation is clearly implied by the sentence.
3. **Resolve all pronouns** such as "it", "he", "she", "they", etc. using the surrounding context. Replace all pronouns in the triple with their correct referents.
   - Do not include any unresolved or ambiguous pronouns in the triples.
   - Be specific and use full entity names instead of pronouns wherever applicable.

4. For each subject and object:
   - Assign a **Subtopic** (a specific category such as "Electronic Musician", "Sound Label", etc.)
   - Assign a **Main topic** (a broader category such as "Music", "Art", etc.)
   - Ensure the subtopic and main topic reflect both the entity and the overall context of the document.

5. Return only valid JSON in the specified format. Do not include markdown, comments, or any other text.
6. Ensure that the JSON is well-formed and valid.

---Examples---

**Document**:
"Inéz is an Estonian new media artist. She lives in Finland. She has composed electronic music."

**Output**:
[
  {
    "triple": ["Inéz", "is", "an Estonian new media artist"],
    "sentence": "Inéz is an Estonian new media artist.",
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
    "triple": ["Inéz", "lives in", "Finland"],
    "sentence": "She lives in Finland.",
    "subject": {
      "subtopic": "New Media Artist",
      "main_topic": "Art"
    },
    "object": {
      "subtopic": "Country",
      "main_topic": "Geography"
    }
  },
  {
    "triple": ["Inéz", "has composed", "electronic music"],
    "sentence": "She has composed electronic music.",
    "subject": {
      "subtopic": "Composer",
      "main_topic": "Music"
    },
    "object": {
      "subtopic": "Genre",
      "main_topic": "Music"
    }
  }
]

---Examples---

**Document**:
"Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer."

**Output**:
[
  {
    "triple": ["Scott Derrickson", "born in", "1966"],
    "sentence": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
    "subject": {
      "subtopic": "Director",
      "main_topic": "Film"
    },
    "object": {
      "subtopic": "Birth Year",
      "main_topic": "Biography"
    }
  },
  {
    "triple": ["Scott Derrickson", "nationality", "America"],
    "sentence": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
    "subject": {
      "subtopic": "Director",
      "main_topic": "Film"
    },
    "object": {
      "subtopic": "Nationality",
      "main_topic": "Culture"
    }
  },
  {
    "triple": ["Scott Derrickson", "occupation", "director"],
    "sentence": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
    "subject": {
      "subtopic": "Director",
      "main_topic": "Film"
    },
    "object": {
      "subtopic": "Profession",
      "main_topic": "Work"
    }
  },
  {
    "triple": ["Scott Derrickson", "occupation", "screenwriter"],
    "sentence": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
    "subject": {
      "subtopic": "Screenwriter",
      "main_topic": "Film"
    },
    "object": {
      "subtopic": "Profession",
      "main_topic": "Work"
    }
  },
  {
    "triple": ["Scott Derrickson", "occupation", "producer"],
    "sentence": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
    "subject": {
      "subtopic": "Producer",
      "main_topic": "Film"
    },
    "object": {
      "subtopic": "Profession",
      "main_topic": "Work"
    }
  }
]

Now, I will remind you of the goal and instructions, and then provide the document to extract triples from.

---Goal---
Extract factual (subject, relation, object) triples from the document and classify the **subject** and **object** into a subtopic and a main topic.

---Instructions---
1. Read the entire document below and extract all factual (subject, relation, object) triples. Each triple must be grounded in a specific sentence from the document.
2. Paraphrasing is acceptable **only** if the relation is clearly implied by the sentence.
3. **Resolve all pronouns** such as "it", "he", "she", "they", etc. using the surrounding context. Replace all pronouns in the triple with their correct referents.
   - Do not include any unresolved or ambiguous pronouns in the triples.
   - Be specific and use full entity names instead of pronouns wherever applicable.

4. For each subject and object:
   - Assign a **Subtopic** (a specific category such as "Electronic Musician", "Sound Label", etc.)
   - Assign a **Main topic** (a broader category such as "Music", "Art", etc.)
   - Ensure the subtopic and main topic reflect both the entity and the overall context of the document.

5. Return only valid JSON in the specified format. Do not include markdown, comments, or any other text.
6. Ensure that the JSON is well-formed and valid.


---Input Document---
{{document}}
"""
