TOPIC_CHOICE_PROMPT = """
Goal:
Select all topics from the provided list that are directly relevant to the user's question.
Choose between {min_topics} and {max_topics} topics.
Return only valid JSON.

Instructions:
1. Use only the topics provided in {{TOPIC_LIST}}.
2. Read the question in {{question}}.
3. Preserve the original topic strings exactly as provided.
4. If too many topics are relevant, keep the most useful ones.
5. If very few topics are relevant, still return at least {min_topics} topics by picking the closest matches.

Output format:
{
  "topics": ["TopicLabel1", "TopicLabel2"]
}

Question:
{{question}}

Allowed topics:
{{TOPIC_LIST}}
"""


def get_topic_choice_prompt() -> str:
    return TOPIC_CHOICE_PROMPT
