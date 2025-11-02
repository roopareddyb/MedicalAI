import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image as AgnoImage
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import numpy as np

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

#-----STREAMLIT SETTINGS-----
st.set_page_config(page_title="Medical Image Parser", layout="centered")
st.title("MedicalAI (Agentic Medical Report AI Analyser)")
st.write("Upload a medical lab image → Extract table → Interpret findings")

# Hyperparameter sliders
st.sidebar.header("Model Settings")
temp_extract = st.sidebar.slider("Temperature (Extractor)", 0.0, 1.0, 0.0, 0.1)
temp_interpret = st.sidebar.slider("Temperature (Interpreter)", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.slider("Max Output Tokens", 512, 8192, 4096, 512)

#-----AGENTS-----

# Agent 1 – STRICT table extractor
extract_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        temperature=temp_extract,
        top_p=0.9,
        top_k=40,
        max_output_tokens=max_tokens
    ),
    markdown=False
)

# Agent 2 – Medical interpreter
interpret_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        temperature=temp_interpret,
        top_p=0.9,
        top_k=40,
        max_output_tokens=max_tokens
    ),
    markdown=False
)

#-----PROMPTS-----

extract_prompt = """
You are a STRICT medical OCR extraction AI for lab reports.

STRICT RULES:
- Output plain text table only (NO pipes, NO markdown, NO bullets)
- Columns in this exact order: Test Name, Result, Unit, Reference Range
- One test = one row only (never split across multiple lines)
- If a test name spans multiple lines in the image, MERGE into one line
- Remove stray symbols like {#, [ ], :, ), -
- If a value is missing, use N/A
- Standardize units (e.g., gm/dL not gm/dll, /uL not /UL)
- Do NOT output section headings like "Absolute Count", "Differential Count", "Platelets"
- Output must look like:

Test Name                          Result    Unit      Reference Range
Haemoglobin                        9.1 L     g/dL      13.0–17.0
Red Cell Distribution Width (RDW)  16.5 H    %         11.6–14.6
Immature Platelet Fraction        2.9       %         0–5

IMPORTANT:
- If formatting breaks, rewrite into a single clean aligned table.
- NEVER output multi-line values.
"""

interpret_prompt = """
You are a medical report interpreter.
Use only extracted table as source.
Do NOT invent lab values.

Output format only:

### Key Findings
- ...

### Summary
- ...

### Clinical Notes
- ...
"""

#-----IMAGE PROCESSING-----

def analyze_image(image_path):
    img = PILImage.open(image_path)
    w, h = img.size
    img = img.resize((500, int(500 * h / w)))

    temp_path = "tmp.png"
    img.save(temp_path)
    agno_image = AgnoImage(filepath=temp_path)

    extracted = extract_agent.run(
        extract_prompt,
        images=[agno_image]
    ).content

    interpreted = interpret_agent.run(
        f"{interpret_prompt}\n\nExtracted Table:\n{extracted}"
    ).content

    os.remove(temp_path)
    return extracted, interpreted

#-----RAG STORE-----
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name, device="cpu")
        self.chunks = []
        self.embeddings = None

    def index_text(self, text, chunk_size=400):
        self.chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        if len(self.chunks) == 0:
            self.embeddings = None
            return
        self.embeddings = self.embedder.encode(self.chunks, convert_to_tensor=True)

    def retrieve(self, query, top_k=3):
        if self.embeddings is None:
            return ""
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(q_emb, self.embeddings)[0]
        top_results = np.argsort(-cos_scores)[:top_k]
        return "\n\n".join([self.chunks[int(i)] for i in top_results])

qna_agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    markdown=True
)

def generate_qna_answer(report_text, conversation, user_question, retrieved_context):
    conv_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in conversation])
    qna_prompt = f"""
You are a clinical assistant. Use ONLY the report text and retrieved context to answer the user's question.

--- Retrieved Context ---
{retrieved_context}

--- Report Text ---
{report_text}

--- Conversation History ---
{conv_text}

Question:
{user_question}

Answer succinctly. If answer isn't in context, say "Not available in report."
"""
    try:
        resp = qna_agent.run(qna_prompt)
        return resp.content
    except Exception as e:
        return f"QnA error: {e}"

#-----SESSION STATE-----
if "analysis_report" not in st.session_state:
    st.session_state.analysis_report = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "rag_store" not in st.session_state:
    st.session_state.rag_store = EmbeddingManager()

#-----MAIN UI-----
uploaded = st.sidebar.file_uploader("Upload medical report image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded, caption="Uploaded Report", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing medical report..."):
            ext = uploaded.type.split("/")[1]
            temp_img = f"upload.{ext}"
            with open(temp_img, "wb") as f:
                f.write(uploaded.getbuffer())

            table, report = analyze_image(temp_img)
            os.remove(temp_img)

            st.subheader("Extracted Table")
            st.code(table)

            st.subheader("Interpreted Medical Summary")
            st.markdown(report, unsafe_allow_html=True)

            st.session_state.analysis_report = report
            st.session_state.rag_store.index_text(report)

else:
    st.info("Upload a report image to begin.")

#-----RAG CHAT-----
if st.session_state.analysis_report:
    st.divider()
    st.subheader("Ask questions about the analyzed report")

    for user_q, ai_a in st.session_state.conversation:
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**Assistant:** {ai_a}")

    user_question = st.text_input("Type a question about the report")

    if user_question:
        retrieved_ctx = st.session_state.rag_store.retrieve(user_question, top_k=3)

        with st.expander("Retrieved context (RAG)"):
            st.markdown(retrieved_ctx if retrieved_ctx.strip() else "_No context retrieved._")

        with st.spinner("Generating answer..."):
            answer = generate_qna_answer(
                st.session_state.analysis_report,
                st.session_state.conversation,
                user_question,
                retrieved_ctx
            )

        st.markdown(f"**Assistant:** {answer}")
        st.session_state.conversation.append((user_question, answer))

        if len(st.session_state.conversation) > 25:
            st.session_state.conversation = st.session_state.conversation[-25:]
