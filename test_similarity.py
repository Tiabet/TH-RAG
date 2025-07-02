import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import dotenv

# 1. OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 2. 문서 및 제목 정의
documents = [
    "In France, the Eiffel Tower is in Paris.",
    "The Great Wall is in China.",
    "The Leaning Tower of Pisa is in Pisa, Italy.",
    "Statue of Liberty stands in New York, United States.",
    "Mount Fuji is the highest mountain in Japan."
]
titles = [
    "Eiffel Tower",
    "Great Wall",
    "Leaning Tower of Pisa",
    "Statue of Liberty",
    "Mount Fuji"
]

# 3. 유저 쿼리 정의
query = "Where is the Iron Lady located?"

# 4. LLM을 이용해 가상의 답변 생성
def generate_hypothetical_answer(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions concisely."},
            {"role": "user", "content": f"Please generate a plausible, concise answer to the question: '{query}'"}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# 가상 답변 생성
# hypo_answer = generate_hypothetical_answer(query)

# 5. 문서, 쿼리, 가상 답변 임베딩
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype="float32")

doc_embeddings = [get_embedding(doc) for doc in documents]
query_embedding = get_embedding(query)
# hypo_embedding = get_embedding(hypo_answer)
hypo_embedding = get_embedding("The Iron Lady is located in Seoul, South Korea.")  # 가상의 답변에 대한 임베딩

# 6. 유사도 계산
def compute_similarities(query_emb, doc_embs):
    return cosine_similarity([query_emb], doc_embs).flatten()

sim_query = compute_similarities(query_embedding, doc_embeddings)
sim_hypo = compute_similarities(hypo_embedding, doc_embeddings)

# 7. 결과 비교
df = pd.DataFrame({
    "Document": titles,
    "Similarity_to_Query": sim_query,
    "Similarity_to_HypoAnswer": sim_hypo
}).sort_values("Similarity_to_HypoAnswer", ascending=False)

print("Query:", query)
# print("Generated Hypothetical Answer:", hypo_answer)
print("\nRetrieval Results:")
print(df)
