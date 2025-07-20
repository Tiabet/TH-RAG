EVALUATION_PROMPT = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    ---Goal---
    You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

    1. **Comprehensiveness**:
        Does the answer address all key parts of the question with sufficient explanation?

    2. **Diversity**:
        Does the answer present different perspectives, nuances, or supporting ideas?

    3. **Empowerment**:
        Does the answer help the reader understand the topic better or make more informed decisions?

    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

    ---Rules---
    - The length of the answers is not a criterion for evaluation. Focus on the quality and depth of the content provided in each answer.
    - Longer answers are penalized if they contain unnecessary or repetitive content

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