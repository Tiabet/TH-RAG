import matplotlib.pyplot as plt
import json
from collections import Counter

# 평가 결과 불러오기
with open("UltraDomain/result/agriculture_judged_results_naive.json", encoding="utf-8") as f:
    data = json.load(f)

categories = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
model_scores = {
    "KG-RAG": Counter(),
    "LightRAG": Counter()
}

# 결과 집계
for item in data:
    if "error" in item:
        continue
    for cat in categories:
        winner = item.get(cat, {}).get("Winner")
        if winner == "Answer 1":
            model_scores[item["answer1_model"]][cat] += 1
        elif winner == "Answer 2":
            model_scores[item["answer2_model"]][cat] += 1

# 시각화 데이터 정리
labels = categories
kgrag_values = [model_scores["KG-RAG"][cat] for cat in labels]
lightrag_values = [model_scores["LightRAG"][cat] for cat in labels]
totals = [k + l for k, l in zip(kgrag_values, lightrag_values)]

kgrag_pct = [k / t * 100 if t > 0 else 0 for k, t in zip(kgrag_values, totals)]
lightrag_pct = [l / t * 100 if t > 0 else 0 for l, t in zip(lightrag_values, totals)]

# 막대 그래프
x = range(len(labels))
bar_width = 0.35
fig, ax = plt.subplots()
ax.bar(x, kgrag_pct, width=bar_width, label='KG-RAG', color='blue')
ax.bar([i + bar_width for i in x], lightrag_pct, width=bar_width, label='Naive', color='orange')

# 퍼센트 텍스트
for i in x:
    ax.text(i, kgrag_pct[i] + 1, f"{kgrag_pct[i]:.1f}%", ha='center')
    ax.text(i + bar_width, lightrag_pct[i] + 1, f"{lightrag_pct[i]:.1f}%", ha='center')

# 설정
ax.set_xlabel("Evaluation Criteria")
ax.set_ylabel("Winning Percentage (%)")
ax.set_title("KG-RAG vs Naive Evaluation Results (agriculture)")
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()
