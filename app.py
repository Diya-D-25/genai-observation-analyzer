import streamlit as st
import requests
import fitz  # PyMuPDF
import docx
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from io import BytesIO

st.set_page_config(page_title="GenAI Consulting Assistant", layout="wide")
st.title("GenAI Consulting Assistant")

# ======================
# API Key Setup
# ======================
together_api_key = st.secrets.get("TOGETHER_API_KEY")
if not together_api_key:
    st.error("❌ Together AI API key not found. Please add it under Streamlit Secrets.")
    st.stop()

# ======================
# Local Storage for Conversation History
# ======================
HISTORY_DIR = "client_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)

def load_history(client_name):
    path = os.path.join(HISTORY_DIR, f"{client_name}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_history(client_name, history):
    path = os.path.join(HISTORY_DIR, f"{client_name}.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

# ======================
# Helper: Analyze with Together AI
# ======================
def analyze_with_together(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {res.status_code} - {res.text}"

# ======================
# Text Extraction
# ======================
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.lower().endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

# ======================
# Chunking
# ======================
def chunk_text(text, chunk_size=5000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# ======================
# Streamlit Tabs
# ======================
tabs = st.tabs(["📄 Document Analysis", "💬 Client Chat", "📊 Incident Logs Analysis"])

# ======================
# TAB 1: Document Analysis
# ======================
with tabs[0]:
    client_name = st.text_input("Enter Client Name for this Analysis")
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"], key="doc_upload")

    if uploaded_file and client_name:
        with st.spinner("Extracting and analyzing document..."):
            text = extract_text(uploaded_file)
            if not text.strip():
                st.error("❌ No readable text found in the document.")
            else:
                merged_table = ""
                for chunk in chunk_text(text):
                    prompt = f"""
You are a senior technology strategy consultant reviewing a document to extract detailed insights across product, engineering, architecture, and operations. 

Your task is to analyze the document deeply and return a **structured table** with the following 4 columns:
- **Category**: Accurately determine the most appropriate category from the following:  
  1. Product Roadmap and Overview  
  2. Architecture & Tech Stack
  3. Operating Model  
  4. SDLC & SMART Practices  
  5. Cloud Infrastructure   
  6. Governance & Decision-Making  
  If none of the above are relevant, create a new fitting category.
  
- **Observation**: Describe the situation or insight clearly and specifically — reflect strategic gaps, inefficiencies, fragmentation, misalignments, or decisions made.

- **Risk**: Go beyond surface risks — analyze **root causes**, **long-term business or operational impact**, and **stakeholders affected**.

- **Recommendation**: Provide a **strong, actionable**, and **consulting-grade** recommendation. Tie it to improvement levers like ROI, cost reduction, operational maturity, architecture optimization, backlog management, etc.

**Instructions**:
- Thoroughly analyze the entire document — don't skip or skim.
- Think like a consultant preparing for a tech due diligence or CXO-level workshop.
- Ensure that **each entry has business or architectural relevance**.
- Tone must be clear, professional, and executive-ready.
- Output only the markdown table, without explanations or headers.

Text to analyze:
{chunk}
"""
                    result = analyze_with_together(prompt)
                    merged_table += result + "\n"

                st.markdown(merged_table)
                history = load_history(client_name)
                history.append({"type": "document_analysis", "content": merged_table})
                save_history(client_name, history)

# ======================
# TAB 2: Client Chat
# ======================
with tabs[1]:
    client_name_chat = st.text_input("Enter Client Name to Continue Chat")
    if client_name_chat:
        history = load_history(client_name_chat)
        for h in history:
            if h["type"] == "chat":
                st.markdown(f"**You:** {h['user']}")
                st.markdown(f"**AI:** {h['assistant']}")

        user_input = st.text_area("Your message")
        if st.button("Send"):
            response = analyze_with_together(user_input)
            st.markdown(f"**AI:** {response}")
            history.append({"type": "chat", "user": user_input, "assistant": response})
            save_history(client_name_chat, history)

# ======================
# TAB 3: Incident Logs Analysis
# ======================
with tabs[2]:
    st.subheader("Upload Incident Logs for Analysis & Charts")
    incident_file = st.file_uploader("Upload CSV/XLSX Incident Logs", type=["csv", "xlsx"], key="incident_upload")

    if incident_file:
        # Read file
        if incident_file.name.endswith(".csv"):
            df = pd.read_csv(incident_file)
        else:
            df = pd.read_excel(incident_file)

        # Auto-calculate "No. of Occurrences" if missing
        if "No. of Occurrences" not in df.columns:
            if "Incident ID" in df.columns:
                df["No. of Occurrences"] = df.groupby("Incident ID")["Incident ID"].transform("count")
            else:
                st.warning("⚠ 'Incident ID' column not found, cannot auto-calculate occurrences.")

        st.write("### Incident Data Preview", df.head())

        # Chart: Most frequent incident types
        if "Type" in df.columns:
            type_counts = df["Type"].value_counts()
            fig, ax = plt.subplots()
            type_counts.plot(kind="bar", ax=ax)
            ax.set_title("Most Frequent Incident Types")
            ax.set_xlabel("Type")
            ax.set_ylabel("Occurrences")
            st.pyplot(fig)

        # Send to Together AI for root cause / risks / recos
        incident_text = df.to_string(index=False)
        prompt = f"""
You are a technology consultant. You are given incident logs in tabular form.  
Return a markdown table with these columns:
Incident ID | Type | No. of Occurrences | Open Date | Close Date | Description | Root Cause | Risk | Recommendation  

Analyze patterns and provide consulting-grade insights, not just surface-level risks.  
Where possible, identify systemic issues or recurring root causes.

Incident Logs:
{incident_text}
"""
        insights = analyze_with_together(prompt)
        st.markdown(insights)
