import streamlit as st
import requests
import fitz  # PyMuPDF
import docx
import pandas as pd
import matplotlib.pyplot as plt
import re

st.title("GenAI Observation, Risk & Incident Analyzer")

together_api_key = st.secrets.get("TOGETHER_API_KEY")
if not together_api_key:
    st.error("❌ Together AI API key not found. Please add it under Streamlit Secrets.")
    st.stop()

# ----------------- Helper: Extract Text -----------------
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.lower().endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.name.lower().endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

# ----------------- Helper: Chunk Text -----------------
def chunk_text(text, chunk_size=3000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# ----------------- Together AI Call -----------------
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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

# ----------------- Incident Log Analysis -----------------
def analyze_incident_logs(df):
    df.columns = [col.strip() for col in df.columns]  # clean column names
    incident_id_col = None
    for col in df.columns:
        if re.search(r"incident.?id", col, re.IGNORECASE):
            incident_id_col = col
            break
    if not incident_id_col:
        st.error("'Incident ID' column not found, cannot auto-calculate occurrences.")
        return

    # Count occurrences
    occ_counts = df[incident_id_col].value_counts().reset_index()
    occ_counts.columns = ["Incident ID", "Occurrences"]

    # Merge back into original data
    df = pd.merge(df, occ_counts, on="Incident ID", how="left")

    # Plot
    st.subheader("Incident Frequency Chart")
    plt.figure(figsize=(8, 4))
    occ_counts.head(10).plot(kind="bar", x="Incident ID", y="Occurrences", legend=False)
    plt.ylabel("Occurrences")
    plt.title("Top 10 Frequent Incidents")
    st.pyplot(plt)

    return df

# ----------------- Streamlit Tabs -----------------
tab1, tab2 = st.tabs(["📄 Client Document Analysis", "📊 Incident Log Analysis"])

# ---- Tab 1: Document Analysis ----
with tab1:
    uploaded_file = st.file_uploader("Upload Client Document", type=["pdf", "docx", "txt"])
    if uploaded_file:
        with st.spinner("Processing document in chunks..."):
            text = extract_text(uploaded_file)
            if not text.strip():
                st.error("❌ No readable text found in the document.")
            else:
                merged_output = ""
                for chunk in chunk_text(text):
                    chunk_result = analyze_text_with_together(chunk, together_api_key)
                    merged_output += chunk_result + "\n"
                st.markdown(merged_output)

# ---- Tab 2: Incident Log Analysis ----
with tab2:
    uploaded_incident_file = st.file_uploader("Upload Incident Log", type=["csv", "xlsx"])
    if uploaded_incident_file:
        if uploaded_incident_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_incident_file)
        else:
            df = pd.read_excel(uploaded_incident_file)

        analyzed_df = analyze_incident_logs(df)
        if analyzed_df is not None:
            st.subheader("Processed Incident Data")
            st.dataframe(analyzed_df)
