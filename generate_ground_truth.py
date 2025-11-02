import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI()

IMG_DIR = "dataset/images"
GT_DIR = "dataset/ground_truth"
os.makedirs(GT_DIR, exist_ok=True)

extract_prompt = """
You are a STRICT medical OCR extraction model.

OUTPUT RULES:
- Plain text table only
- Columns: Test Name, Result, Unit, Reference Range
- One test per line
- Merge multiline test names
- No pipes, no bullets, no markdown
- If unclear write N/A
"""

interpret_prompt = """
You are a medical interpretation model interpreting lab reports.

Output:
### Key Findings
- ...

### Summary
- ...

### Clinical Notes
- ...

### Model Confidence (0-100%)
"""

def encode_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_gt(img_path):
    img_b64 = encode_image(img_path)

    # Extract table
    extract_resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": extract_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract medical lab table"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ]
    )
    table = extract_resp.choices[0].message.content.strip()

    # Interpret report
    interpret_resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": interpret_prompt},
            {"role": "user", "content": f"Extracted Table:\n{table}"}
        ]
    )
    interpretation = interpret_resp.choices[0].message.content.strip()

    return table, interpretation

def main():
    for file in os.listdir(IMG_DIR):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(IMG_DIR, file)
        gt_path = os.path.join(GT_DIR, os.path.splitext(file)[0] + ".txt")

        if os.path.exists(gt_path):
            print(f"✅ GT exists, skipping {file}")
            continue

        print(f"📄 Generating GT for {file}...")

        try:
            table, interpretation = generate_gt(img_path)

            with open(gt_path, "w", encoding="utf-8") as f:
                f.write("### Extracted Table\n")
                f.write(table + "\n\n")
                f.write("### Interpretation\n")
                f.write(interpretation)

            print(f"✅ GT saved: {gt_path}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

if __name__ == "__main__":
    main()
