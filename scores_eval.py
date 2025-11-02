import pandas as pd

RESULTS_FILE = "dataset/results/scores.csv"
PRED_DIR = "dataset/results/predictions"
df = pd.read_csv(RESULTS_FILE)

agg = {
    "Mean BLEU": df["BLEU"].mean(),
    "Mean ROUGE-1": df["ROUGE-1"].mean(),
    "Mean ROUGE-2": df["ROUGE-2"].mean(),
    "Mean ROUGE-L": df["ROUGE-L"].mean()
}

df.to_csv(RESULTS_FILE, index=False)

print("\nEvaluation Completed")
print(f"Saved extracted tables & interpretations in: {PRED_DIR}")
print(f"Scores file: {RESULTS_FILE}\n")

print("=== Dataset-Level Aggregated Scores ===")
print(pd.DataFrame([agg]))