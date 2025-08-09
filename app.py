import streamlit as st
import requests
import fitz  # PyMuPDF
import docx
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------- CONFIG -----------------
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("❌ Together AI API key not found. Please add it under Streamlit Secrets.")
    st.stop()

# ----------------- FUNCTIONS -----------------
def load_history(client_name):
    filepath = os.path.join(HISTORY_DIR, f"{client_name}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {"profile": "", "analyses": []}

def save_history(client_name, history):
    filepath = os.path.join(HISTORY_DIR, f"{client_name}.json")
    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)

def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.lower().endswith(".docx"):
        docx_doc = docx.Document(file)
        return "\n".join(p.text for p in docx_doc.paragraphs)
    elif file.name.lower().endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

def chunk_text(text, max_chars=10000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

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
        "max_tokens": 1500
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {res.status_code} - {res.text}"

def analyze_incident_logs(df):
    insights = {}
    if 'type' in df.columns:
        insights['by_type'] = df['type'].value_counts()
    if 'open_date' in df.columns and 'close_date' in df.columns:
        df['open_date'] = pd.to_datetime(df['open_date'], errors='coerce')
        df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
        df['resolution_time'] = (df['close_date'] - df['open_date']).dt.days
        insights['avg_resolution_time'] = df['resolution_time'].mean()
    if 'root_cause' in df.columns:
        insights['by_root_cause'] = df['root_cause'].value_counts()
    return insights

def plot_bar(data, title):
    fig, ax = plt.subplots()
    data.plot(kind='bar', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_pie(data, title):
    fig, ax = plt.subplots()
    data.plot(kind='pie', ax=ax, autopct='%1.1f%%')
    ax.set_ylabel('')
    ax.set_title(title)
    st.pyplot(fig)

# ----------------- STREAMLIT UI -----------------
st.title("📊 Client Consulting Assistant")

client_name = st.text_input("Enter Client Name")
if not client_name:
    st.stop()

history = load_history(client_name)

tabs = st.tabs(["Client Profile", "Document Analysis", "Incident Logs Analysis"])

with tabs[0]:
    st.subheader("🗂 Client Profile")
    profile_text = st.text_area("Client Profile Notes", value=history.get("profile", ""), height=200)
    if st.button("💾 Save Profile"):
        history["profile"] = profile_text
        save_history(client_name, history)
        st.success("Profile saved!")

with tabs[1]:
    st.subheader("📄 Document Analysis")
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        st.info(f"Document split into {len(chunks)} chunks for analysis.")

        merged_results = ""
        for idx, chunk in enumerate(chunks, 1):
            st.write(f"🔍 Analyzing chunk {idx}...")
            result = analyze_text_with_together(chunk, TOGETHER_API_KEY)
            merged_results += result + "\n"

        st.markdown(merged_results)
        history["analyses"].append({"date": str(datetime.now()), "result": merged_results})
        save_history(client_name, history)

with tabs[2]:
    st.subheader("📉 Incident Logs Analysis")
    log_file = st.file_uploader("Upload Incident Logs (CSV or XLSX)", type=["csv", "xlsx"])
    if log_file:
        if log_file.name.endswith(".csv"):
            df = pd.read_csv(log_file)
        else:
            df = pd.read_excel(log_file)

        st.write(df.head())
        insights = analyze_incident_logs(df)

        if 'by_type' in insights:
            st.write("**Incidents by Type**")
            plot_bar(insights['by_type'], "Incidents by Type")

        if 'avg_resolution_time' in insights:
            st.write(f"**Average Resolution Time:** {insights['avg_resolution_time']:.2f} days")

        if 'by_root_cause' in insights:
            st.write("**Root Cause Distribution**")
            plot_pie(insights['by_root_cause'], "Root Cause Distribution")
