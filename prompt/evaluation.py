EVALUATION_PROPMPT = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    ---Goal---
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
    - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
    - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

    Here is the question: {query}

    Here are the two answers:
    **Answer 1:**
    {answer1}

    **Answer 2:**
    {answer2}

    Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.
    Please output ONLY the valid JSON below, with no explanation or markdown formatting.
    Output your evaluation in the following JSON format:
    {{
      "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]",  "Explanation": "[Provide explanation here]" }},
      "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
      "Empowerment": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
      "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"  
      }}
    """