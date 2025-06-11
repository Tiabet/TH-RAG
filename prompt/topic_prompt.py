TOPIC_PROMPT = """
You are an assistant that generates concise topic suggestions.
Given the user's question, produce exactly five broad topics relevant to the question.
For each topic, list ten concrete subtopics. Respond in JSON format:
{
  "topics": [
    {"topic": "Topic name", "subtopics": ["Sub1", "Sub2", ...]},
    ... (total five items)
  ]
}
---Instructions---
1. Each topic should be a single word or a short phrase.
2. Each Subtopic should be related to the topic and be a single word or a short phrase.
3. Ensure the response is valid JSON.
Question: {{question}}
"""

if __name__ == "__main__":
    print(TOPIC_PROMPT)
