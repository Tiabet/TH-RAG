TOPIC_PROMPT = """
You are an assistant that generates concise topic suggestions.
Given the user's question, produce exactly five broad topics relevant to the question.
For each topic, list ten concrete subtopics. Respond in JSON format:

---Instructions---
1. Each topic should be a single word or a short phrase.
2. Each Subtopic should be related to the topic and be a single word or a short phrase.
3. Return only valid JSON in the specified format. Do not include markdown, comments, or any other text.
4. Do not include any additional text or explanations outside the JSON format.
5. Ensure that the JSON is well-formed and valid.

---Output Format---
Respond ONLY with a valid JSON array using the following format:

{
  "topics": [
    {"topic": "Topic name", "subtopics": ["Sub1", "Sub2", ...]},
  ]
}

Question: {{question}}
"""

if __name__ == "__main__":
    print(TOPIC_PROMPT)
