SUBTOPIC_CHOICE_PROMPT = """

--- Goal ---
Given the user's question, choose **all** topics from the supplied list that are directly relevant to answering the question.
For the given topic **{TOPIC_LABEL}**, choose every subtopic from the list below that is helpful for answering the user's question.
Select **{min_subtopics} to {max_subtopics}** subtopics. Do **NOT** invent new subtopics.
Always return at least {min_subtopics} subtopics, unless case of list is shorter than {min_subtopics}.

--- Instructions ---
1. Consider only the subtopics provided in **{SUBTOPIC_LIST}**.
2. Read the user's question provided in **{question}**.
3. Output your selection as valid JSON **without** markdown, comments, or extra text.
4. Preserve the original order of **{SUBTOPIC_LIST}** when listing the chosen subtopics.
5. Output JSON Format:
   {"subtopics": ["SubLbl1", "SubLbl2", ...]}
6. You MUST ONLY choose from the list provided below. Do not invent or rephrase any subtopics.
7. If you cannot find any relevant topics, just find the most relevant {min_subtopics} topics.


Question: {question}

--- Allowed Subtopics for {{TOPIC_LABEL}} ---
{{SUBTOPIC_LIST}}

---Remind of Instructions---
--- Instructions ---
1. Consider only the subtopics provided in **{SUBTOPIC_LIST}**.
2. Read the user's question provided in **{question}**.
3. Output your selection as valid JSON **without** markdown, comments, or extra text.
4. Preserve the original order of **{SUBTOPIC_LIST}** when listing the chosen subtopics.
5. Output JSON Format:

   {"subtopics": ["SubLbl1", "SubLbl2", ...]}
6. You MUST ONLY choose from the list provided below. Do not invent or rephrase any subtopics.
7. If you cannot find any relevant subtopics, just find the most relevant {min_subtopics} subtopics.

Question: {question}

--- Allowed Subtopics for {{TOPIC_LABEL}} ---
{{SUBTOPIC_LIST}}

"""
