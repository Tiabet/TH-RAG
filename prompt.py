"""Prompt for extracting a knowledge graph from text.

Running this module will print the prompt text.
"""

PROMPT = """
You are an expert in information extraction and knowledge organization.
Given the following English document, perform the following tasks:

Extract object-relation-object triples from the text. Each triple should represent a factual relationship stated or implied in the text. Use the format (subject, relation, object). Don't use python, just read the document.

For each object and subject in the extracted triples, identify:

Its subtopic: a more specific category or concept it belongs to.

Its main topic: a broader thematic area it belongs to.

Present the results in the following structured format:

Triple: (Subject, Relation, Object)  
Sentence: "Exact sentence from which the triple was extracted."  
Subject → Subtopic: [ ], Main topic: [ ]  
Object → Subtopic: [ ], Main topic: [ ]

Here is the document: 

"""

if __name__ == "__main__":
    print(PROMPT)
