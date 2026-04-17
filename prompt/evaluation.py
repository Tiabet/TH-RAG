EVALUATION_PROMPT = """
You are an expert evaluator comparing two answers to the same question.

Evaluate the answers on these criteria:
1. Comprehensiveness: Does the answer cover the important parts of the question?
2. Diversity: Does the answer include useful nuance, perspective, or supporting detail?
3. Empowerment: Does the answer leave the reader better informed or better able to act?

Question:
{query}

Answer 1:
{answer1}

Answer 2:
{answer2}

Rules:
- Focus on quality, not length.
- Longer answers should not win if they are repetitive or unnecessary.
- Return only valid JSON and no extra text.

Output format:
{{
  "Comprehensiveness": {{"Winner": "Answer 1 or Answer 2", "Explanation": "..."}},
  "Diversity": {{"Winner": "Answer 1 or Answer 2", "Explanation": "..."}},
  "Empowerment": {{"Winner": "Answer 1 or Answer 2", "Explanation": "..."}},
  "Overall Winner": {{"Winner": "Answer 1 or Answer 2", "Explanation": "..."}}
}}
"""
