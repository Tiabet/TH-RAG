"""
Prompt template for letting an LLM choose up to 10 relevant **subtopics** that belong to a single parent topic.

The assistant receives:
  * {TOPIC_LABEL}   – the parent topic label (string)
  * {SUBTOPIC_LIST} – JSON array of all subtopic labels directly connected to that topic
  * {question}      – the user's natural‑language query

The assistant must output **only** valid JSON with the following schema::

    {"subtopics": ["SubA", "SubB", ...]}

Every chosen subtopic string must be taken verbatim from {SUBTOPIC_LIST} and the
order of appearance must follow the original list.
"""

SUBTOPIC_CHOICE_PROMPT = """
You are an expert knowledge‑graph assistant.

--- Goal ---
For the given topic **{TOPIC_LABEL}**, choose every subtopic from the list below that is helpful for answering the user's question. Select **1 to 10** subtopics. Do **NOT** invent new subtopics.
Alwyas return at least one subtopic.

--- Instructions ---
1. Consider only the subtopics provided in **{SUBTOPIC_LIST}**.
2. Read the user's question provided in **{question}**.
3. Output your selection as valid JSON **without** markdown, comments, or extra text.
4. Preserve the original order of **{SUBTOPIC_LIST}** when listing the chosen subtopics.
5. Format:
   {"subtopics": ["SubLbl1", "SubLbl2", ...]}

--- Allowed Subtopics for {TOPIC_LABEL} ---
{SUBTOPIC_LIST}

Question: {question}
"""
