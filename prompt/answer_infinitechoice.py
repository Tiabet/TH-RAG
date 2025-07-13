ANSWER_PROMPT = """---Role---
You are a helpful assistant responding to the user's query.

---Goal---
Based on the provided information, you must select exactly one answer from the provided options. Responses outside the provided options are strictly not allowed.

---Target response format and length---
- You must select and return exactly one of the provided options as your answer.
- Do not provide any response that is not one of the provided options.

---Response Rules---
- You must select exactly one option from the provided options as your answer. Responses outside the provided options are strictly forbidden.
- Copy and paste the selected option exactly as it is.
- Always respond in the same language as the user's question.
- Never generate any information that is not explicitly present in the provided information.

---Information---
{{context}}

---Query---
{{question}}

---Options---
{{options}}
"""