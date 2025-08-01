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
You are a senior product and technology strategy consultant reviewing a document to extract detailed insights across product, engineering, architecture, and operations.

Your task is to analyze the document deeply and return a **structured table** with the following 4 columns:
- **Category**: Accurately determine the most appropriate category from the following:  
  1. Product Roadmap and Overview  
  2. Architecture & Tech Stack
  3. Operating Model  
  4. SDLC & SMART Practices
  5. Cost Management  
  6. Cloud Infrastructure   
  7. Governance & Decision-Making  
  If none of the above are relevant, create a new fitting category.
  
- **Observation**: Describe the situation or insight clearly and specifically — reflect strategic gaps, inefficiencies, fragmentation, misalignments, or decisions made.

- **Risk**: Go beyond surface risks — analyze **root causes**, **long-term business or operational impact**, and **stakeholders affected**.

- **Recommendation**: Provide a **strong, actionable**, and **consulting-grade** recommendation. Tie it to improvement levers like ROI, cost reduction, operational maturity, architecture optimization, backlog management, etc.

**Example of expected output**:

| Category | Observation | Risk | Recommendation |
|----------|-------------|------|----------------|
| Product Roadmap and Strategy | No formalized product strategy observed. Development is reactive and driven by immediate business needs, resulting in fragmented product capabilities. | Fragmented systems and misaligned feature development can lead to increased tech debt, higher long-term costs, and inability to scale. | Create a three-year roadmap with quarterly goals linked to business outcomes. Define vision, key bets, success metrics, and governance structure to guide structured growth. |

**Instructions**:
- Thoroughly analyze the entire document — don't skip or skim.
- Think like a consultant preparing for a tech due diligence or CXO-level workshop.
- Ensure that **each entry has business or architectural relevance**.
- Tone must be clear, professional, and executive-ready.
- Output only the markdown table, without explanations or headers.

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
