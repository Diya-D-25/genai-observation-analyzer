import streamlit as st
import requests
import fitz  # PyMuPDF
import docx

st.title("GenAI Observation & Risk Extractor")

together_api_key = st.secrets.get("TOGETHER_API_KEY")

if not together_api_key:
    st.error("❌ Together AI API key not found. Please add it under Streamlit Secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.lower().endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def analyze_text_with_together(text, api_key):
    prompt = f"""
You are a senior technology and product strategy consultant. Carefully read the document below, and extract well-thought-out insights that would help assess the maturity, quality, and risks associated with the product, technology, processes, or operations described.

Your task is to produce a **structured table** with the following columns:

- **Category**: Identify the right category for each insight based on the content (e.g., Architecture, Infrastructure, Tech Stack  DevOps, Product Roadmap, Integration, Data, Cost Management, Operating Model, etc.). You are free to define categories — do not limit yourself to predefined ones.
- **Observation**: A specific, clearly stated insight derived from the content. Think critically — go beyond surface-level facts and highlight patterns, inefficiencies, or notable decisions.
- **Associated Risk**: Write a **detailed explanation** of the risk or downside associated with the observation. Go deep — explain the **root cause**, **potential impact**, **which stakeholder it affects**, and **how it could evolve if not addressed**. Avoid one-word or generic risks. This could include strategic misalignment, scalability issues, technical debt, unclear ownership, or dependency risks.
- **Recommendation**: Provide a concise, practical, and executive-level recommendation to address the risk or improve the current state.

**Additional guidelines**:
- Read the full document deeply and extract as many relevant insights as possible — do not be brief or superficial.
- Ensure your tone is professional, analytical, and suitable for a consulting or due diligence report.
- Use a markdown table for the output with the headers: `Category | Observation | Risk | Recommendation`.
- Only include meaningful insights — avoid generic or vague entries.
- Aim for both quality and quantity, do not omit valid observations.

Text to analyze:
{text}
"""
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {res.status_code} - {res.text}"

if uploaded_file:
    with st.spinner("Extracting and analyzing..."):
        text = extract_text(uploaded_file)
        if not text.strip():
            st.error("❌ No readable text found in the document.")
        else:
            output = analyze_text_with_together(text, together_api_key)
            st.markdown(output)
