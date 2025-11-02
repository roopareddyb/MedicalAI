import os
import time
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image as AgnoImage
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# ========= Paths =========
IMG_DIR = "dataset/images"
GT_DIR = "dataset/ground_truth"
PRED_DIR = "dataset/results/predictions"
RESULTS_FILE = "dataset/results/scores.csv"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs("dataset/results", exist_ok=True)

# ========= Models =========
extract_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        temperature=0.0,
        top_p=0.9,
        top_k=40, 
        max_output_tokens=4096
    ),
    markdown=False
)

interpret_agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash",
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        max_output_tokens=4096
    ),
    markdown=False
)

# ========= Prompts =========
extract_prompt = """
You are a STRICT medical OCR extraction AI for lab reports.

STRICT RULES:
- Output plain text table only (NO pipes, NO markdown, NO bullets)
- Columns in this exact order: Test Name, Result, Unit, Reference Range
- One test = one row
- Merge wrapped names into one line
- Standardize units (g/dL, /uL, etc.)
- Use N/A when missing
- Ignore section headings like 'Absolute Count', 'Platelets'
"""

interpret_prompt = """
You are a medical report interpreter.
Use only the extracted table. Do NOT invent values.

Output:

### Key Findings
- ...

### Summary
- ...

### Clinical Notes
- ...

### Model Confidence (0-100%)
"""

# ========= Helpers =========
def analyze_image(path):
    img = PILImage.open(path)
    w, h = img.size
    img = img.resize((500, int(500 * h / w)))
    tmp = "tmp_eval.png"
    img.save(tmp)

    extracted = extract_agent.run(extract_prompt, images=[AgnoImage(filepath=tmp)]).content
    interpreted = interpret_agent.run(f"{interpret_prompt}\n\nExtracted Table:\n{extracted}").content

    os.remove(tmp)
    return extracted.strip(), interpreted.strip()

def evaluate(gt, pred):
    ref_tokens = [gt.split()]
    pred_tokens = pred.split()
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)

    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge = scorer.score(gt, pred)

    return bleu, rouge

# ========= Main Loop =========
results = []

for file in os.listdir(IMG_DIR):
    if not file.lower().endswith(("jpg","jpeg","png")):
        continue

    base = os.path.splitext(file)[0]
    img_path = os.path.join(IMG_DIR, file)
    gt_path = os.path.join(GT_DIR, base + ".txt")

    extracted_out = os.path.join(PRED_DIR, base + "_extracted.txt")
    interpreted_out = os.path.join(PRED_DIR, base + "_interpreted.txt")

    print(f"Processing {file} ...")

    extracted, interpreted = analyze_image(img_path)

    formatted_output = (
        "### Extracted Table\n"
        + extracted
        + "\n\n### Interpretation\n"
        + interpreted
    )

    # Save formatted extracted + interpreted output
    with open(extracted_out, "w", encoding="utf-8") as f:
        f.write(formatted_output)

    with open(interpreted_out, "w", encoding="utf-8") as f:
        f.write(formatted_output)

    if not os.path.exists(gt_path):
        print(f"Missing GT: {gt_path}, skipping.")
        continue

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()

    # Evaluate EXTRACTED table vs GT table
    bleu, rouge = evaluate(gt_text, extracted)

    results.append({
        "file": file,
        "BLEU": bleu,
        "ROUGE-1": rouge["rouge1"].fmeasure,
        "ROUGE-2": rouge["rouge2"].fmeasure,
        "ROUGE-L": rouge["rougeL"].fmeasure
    })

    time.sleep(5)  # prevent rate limiting

df = pd.DataFrame(results)

print("\nDONE! Extraction + Evaluation Completed")
print(f"Saved extracted tables & interpretations in: {PRED_DIR}")
print(f"Scores file: {RESULTS_FILE}\n")

