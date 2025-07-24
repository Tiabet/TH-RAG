TOPIC_CHOICE_PROMPT = """

--- Goal ---
Given the user's question, choose **all** topics from the supplied list that are directly relevant to answering the question.
Select **between {min_topics} and {max_topics}** topics. Choose exhaustively but do **NOT** invent new topics.
**Return the chosen topics *exactly* as they appear in the list.**
Always return at least {min_topics} topics.

--- Instructions ---
1. The list of allowed topics will be provided in the placeholder **{TOPIC_LIST}**.
2. Read the user question provided in the placeholder **{question}**.
3. Identify every topic from **{TOPIC_LIST}** that is pertinent to the question.
4. Output **only** valid JSON. Do **not** include markdown, comments, or extra text.
5. Output JSON format:

{
  "topics": ["TopicLabel1", "TopicLabel2", ...]
}

6. You MUST ONLY choose from the list provided below. **Do not invent or rephrase any subtopics.**
7. If you cannot find any relevant topics, just find the most relevant {min_topics} topics.

Question: {{question}}

--- Allowed Topics ---
{{TOPIC_LIST}}

---Remind of Instructions---

--- Instructions ---
1. The list of allowed topics will be provided in the placeholder **{TOPIC_LIST}**.
2. Read the user question provided in the placeholder **{question}**.
3. Identify every topic from **{TOPIC_LIST}** that is pertinent to the question.
4. Output **only** valid JSON. Do **not** include markdown, comments, or extra text.
5. Output JSON format:

{
  "topics": ["TopicLabel1", "TopicLabel2", ...]
}

6. You MUST ONLY choose from the list provided below. **Do not invent or rephrase any subtopics.**
7. If you cannot find any relevant topics, just find the most relevant {min_topics} topics.

"""
