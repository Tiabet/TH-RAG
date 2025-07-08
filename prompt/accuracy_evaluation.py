ACCURACY_EVALUATION_PROMPT = """
Instructions
You will receive a question (referred to as Query), a ground truth answer (referred to as Answer), and a model-generated answer (referred to as Response).
Your task is to compare the Answer and Response and determine whether they align.

Note: The ground truth answer may sometimes be embedded within the model-generated answer.
You need to carefully analyze and discern whether they align.

Your Output:
If the two answers align, respond with yes.
If they do not align, respond with no.
If you are very uncertain, respond with unclear.
Your response should first include yes, no, or unclear, followed by an explanation.

Example 1  
Query: Which NBA team drafted Jason Collins in 2001?  
Answer: Houston Rockets  
Response: The basketball player who was drafted 18th overall in 2001 is Jason Collins, who was selected by the Houston Rockets.  
Expected output: yes

Example 2  
Query: Are Variety and The Advocate both LGBT-interest magazines?  
Answer: no  
Response: Yes, both Variety and The Advocate are LGBT-interest magazines. The Advocate is explicitly identified as an American LGBT-interest magazine, while Variety, although primarily known for its coverage of the entertainment industry, also addresses topics relevant to the LGBT community.  
Expected output: no

Input Data Format  
Query: {query}  
Ground Truth Answer: {answer}  
Model Generated Answer: {response}

Expected Output  
yes, no, or unclear  
An explanation of your choice.

Output:
"""
