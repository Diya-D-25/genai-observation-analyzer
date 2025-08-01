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
You are an experienced technology consultant reviewing a project document or stakeholder input file. Your task is to extract insights that will help assess the quality, risks, and maturity of the product or technology landscape described.

Please analyze the entire text carefully and return the following:

- **Category**: Identify appropriate categories on your own (e.g., Architecture, Infrastructure, Data Strategy, DevOps, Product Roadmap, Cost Management, Tech Stack, Integration, Security, etc.) based on the content — do NOT limit yourself to predefined categories.
- **Observation**: A concise yet specific statement summarizing a key finding or fact from the document.
- **Associated Risk**: Any potential issue, inefficiency, or gap linked to the observation.
- **Recommendation**: A clear, actionable suggestion to address the risk or improve the current state.

**Return the output in a clean markdown table** with the following columns:

`Category | Observation | Risk | Recommendation`

Ensure:
- You go through the text thoroughly.
- Output is written in a professional tone, suitable for inclusion in an executive-level consulting report.
- Capture as many relevant rows as needed — do not limit yourself to only a few.

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
