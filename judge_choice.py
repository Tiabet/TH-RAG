import json
import pandas as pd
from sklearn.metrics import accuracy_score
# hybrid_infinitechoice_result
# Load the two JSON files
with open("InfiniteChoice/result/hybrid_infinitechoice_result.json", "r", encoding="utf-8") as f1, open("InfiniteChoice/qa.json", "r", encoding="utf-8") as f2:
    correct_answers = json.load(f1)
    predicted_answers = json.load(f2)

# Convert to DataFrames for easy comparison
df_correct = pd.DataFrame(correct_answers)
df_predicted = pd.DataFrame(predicted_answers)

# Merge on query to align answers
df_merged = pd.merge(df_correct, df_predicted, on="query", suffixes=('_correct', '_predicted'))

# Calculate accuracy
accuracy = accuracy_score(df_merged["answer_correct"], df_merged["answer_predicted"])

print(accuracy)