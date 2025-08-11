# app.py
import streamlit as st
import requests
import fitz  # PyMuPDF
import docx
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import os
import json
import re
from datetime import datetime

st.set_page_config(page_title="GenAI Consulting Assistant", layout="wide")

# -----------------------
# Config / API Key
# -----------------------
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("❌ Together AI API key not found. Please add it under Streamlit Secrets.")
    st.stop()

HISTORY_FILE = "history.json"
os.makedirs("client_histories", exist_ok=True)

# -----------------------
# Utilities: history
# -----------------------
def load_all_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_all_history(all_hist):
    with open(HISTORY_FILE, "w") as f:
        json.dump(all_hist, f, indent=2, default=str)

def load_client_history(client_name):
    all_hist = load_all_history()
    return all_hist.get(client_name, {"chats": [], "analyses": []})

def save_client_history(client_name, client_history):
    all_hist = load_all_history()
    all_hist[client_name] = client_history
    save_all_history(all_hist)

# -----------------------
# Helpers: text extraction
# -----------------------
def extract_text_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            uploaded_file.seek(0)
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        if name.endswith(".docx"):
            uploaded_file.seek(0)
            docx_doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in docx_doc.paragraphs)
        if name.endswith(".txt"):
            uploaded_file.seek(0)
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return ""

# -----------------------
# Helpers: chunking
# -----------------------
def chunk_text_words(text, chunk_words=3000, overlap_words=200):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_words]
        chunks.append(" ".join(chunk))
        i += chunk_words - overlap_words
    return chunks

def chunk_dataframe_rows(df, batch_rows=200):
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start : start + batch_rows]

# -----------------------
# Together AI wrapper
# -----------------------
def call_together(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=1500, temp=0.2):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
    except Exception as e:
        return f"❌ Error: request failed: {e}"
    if r.status_code == 200:
        try:
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            return f"❌ Error: unexpected response format: {r.text}"
    else:
        return f"❌ Error: {r.status_code} - {r.text}"

# -----------------------
# Prompts (original ones retained)
# -----------------------
DOCUMENT_PROMPT_TEMPLATE = """
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

INCIDENT_PROMPT_TEMPLATE = """
You are a technology operations analyst. Given the incident rows below, produce a markdown table with columns:

Incident ID | Type | No. of Occurrences | Open Date | Close Date | Description | Root Cause | Risk | Recommendation

Tasks:
- Group rows by Type or Incident ID when appropriate.
- For each group, list Incident IDs concatenated (comma-separated), count occurrences, earliest Open Date, latest Close Date.
- For Description, synthesize a concise summary of the incidents.
- For Root Cause, infer likely causes from descriptions.
- For Risk, explain business/operational impact.
- For Recommendation, provide prioritized actions (90-day short-term and medium-term).

Return only the markdown table.

Incident rows:
{table_text}
"""

# -----------------------
# Additional prompts & constants (new)
# -----------------------
ENHANCE_FINDINGS_PROMPT = """
You are a senior consulting analyst. Given the following findings table, add the following columns:
- Severity (High, Medium, Low)
- Time Horizon (Immediate, Short-term, Medium-term, Long-term)
- Priority Type (Quick Win / Strategic)
- Business KPIs (up to 3 relevant KPIs per recommendation, e.g., ROI, Operational Cost Reduction, Time-to-Market, Customer Satisfaction, Scalability, Risk Reduction)

Keep all original columns, append the new ones at the end.
Return only the enhanced markdown table.
Table:
{table}
"""

CORRELATION_PROMPT = """
You are a senior consultant. Given the following datasets:
1. Document Findings
2. Incident Log Insights
Identify:
- Overlapping or related issues across datasets
- Risks corroborated by multiple sources
- Recommendations that address multiple issues simultaneously
Return a concise markdown table: Theme | Linked Risks | Consolidated Recommendation | Impact
Document Findings:
{doc_findings}

Incident Insights:
{incident_findings}
"""

# Simple benchmark examples (extend as needed)
BENCHMARKS = {
    "dev_to_qa": (5, 1),  # dev:qa = 5:1
    "pm_ba_dev_qa": (1, 1, 8, 2)  # pm:ba:dev:qa
}

# -----------------------
# Helper: parse markdown table (moveable so other blocks can use it)
# -----------------------
def parse_markdown_table(md_text):
    rows = [r for r in md_text.splitlines() if r.strip() != ""]
    header_idx = None
    for i, r in enumerate(rows):
        if "|" in r and re.search(r"[A-Za-z]", r):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()
    header = [c.strip() for c in rows[header_idx].split("|") if c.strip() != ""]
    data_start = header_idx + 1
    if data_start < len(rows) and re.match(r'^\s*\|?\s*-+', rows[data_start]):
        data_start += 1
    data_rows = []
    for r in rows[data_start:]:
        # handle lines that may include | inside columns by limiting split
        parts = [c.strip() for c in r.split("|") if c.strip() != ""]
        if len(parts) == len(header):
            data_rows.append(parts)
    if not data_rows:
        return pd.DataFrame()
    df = pd.DataFrame(data_rows, columns=header)
    return df

# -----------------------
# UI
# -----------------------
st.title("GenAI Consulting Assistant")

client_name = st.text_input("Client name (used to store history)", value="", max_chars=80)
if not client_name:
    st.info("Enter a client name to start. History and analyses will be saved locally.")
    st.stop()

client_history = load_client_history(client_name)

tabs = st.tabs(["💬 Client Chat", "📄 Document Analysis", "📊 Incident Logs"])

# -----------------------
# TAB: Client Chat
# -----------------------
with tabs[0]:
    st.header("Client Chat (persistent per client)")
    st.subheader("Conversation history")
    for item in client_history.get("chats", []):
        role = item.get("role", "user")
        if role == "user":
            st.markdown(f"**You:** {item.get('content')}")
        else:
            st.markdown(f"**AI:** {item.get('content')}")

    st.subheader("Send a message")
    user_msg = st.text_area("Message", value="", height=120)
    include_context = st.checkbox("Include recent analyses (as context) in AI prompt", value=True)

    if st.button("Send message"):
        context_text = ""
        if include_context:
            last_analyses = client_history.get("analyses", [])[-3:]
            if last_analyses:
                context_text = "\n\n".join([f"### Analysis ({a.get('ts')}):\n{a.get('result')}" for a in last_analyses])
        prompt = user_msg
        if context_text:
            prompt = f"{context_text}\n\nUser question:\n{user_msg}"

        ai_resp = call_together(prompt)
        st.markdown(f"**AI:** {ai_resp}")
        client_history.setdefault("chats", []).append({"role": "user", "content": user_msg, "ts": str(datetime.utcnow())})
        client_history.setdefault("chats", []).append({"role": "assistant", "content": ai_resp, "ts": str(datetime.utcnow())})
        save_client_history(client_name, client_history)

# -----------------------
# TAB: Document Analysis (modified to support multiple uploads + enhancement)
# -----------------------
with tabs[1]:
    st.header("Document Analysis")
    # allow multiple uploads (added functionality)
    uploaded_files = st.file_uploader("Upload one or more documents (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        all_findings = []
        merged_texts_for_history = []
        for uploaded in uploaded_files:
            with st.spinner(f"Extracting text from {uploaded.name}..."):
                text = extract_text_file(uploaded)
            if not text or not text.strip():
                st.warning(f"No readable text found in {uploaded.name}.")
                continue

            chunks = chunk_text_words(text, chunk_words=3000, overlap_words=200)
            st.info(f"{uploaded.name}: split into {len(chunks)} chunks. Processing sequentially.")
            merged_lines = []
            header_seen = False
            for idx, chunk in enumerate(chunks, start=1):
                st.write(f"Analyzing {uploaded.name} — chunk {idx}/{len(chunks)}...")
                prompt = DOCUMENT_PROMPT_TEMPLATE.format(text=chunk)
                resp = call_together(prompt, max_tokens=1500)
                if resp.startswith("❌ Error"):
                    st.error(resp)
                    break
                lines = [ln for ln in resp.splitlines() if ln.strip() != ""]
                if not header_seen:
                    merged_lines.extend(lines)
                    header_seen = True
                else:
                    # skip repeated header rows in subsequent chunks
                    if len(lines) >= 2 and re.match(r'^\s*\|?-+', lines[1]):
                        rows = [l for i_l, l in enumerate(lines) if i_l >= 2]
                        merged_lines.extend(rows)
                    else:
                        if lines and merged_lines and lines[0].strip() == merged_lines[0].strip():
                            merged_lines.extend(lines[1:])
                        else:
                            merged_lines.extend(lines)

            merged_text = "\n".join(merged_lines)
            merged_texts_for_history.append({"name": uploaded.name, "text": merged_text})
            st.subheader(f"Merged Findings for {uploaded.name} (markdown table)")
            st.markdown(merged_text)

            df_findings = parse_markdown_table(merged_text)
            if not df_findings.empty:
                df_findings["Source Document"] = uploaded.name
                all_findings.append(df_findings)

        if all_findings:
            # combined findings across uploaded files
            df_combined = pd.concat(all_findings, ignore_index=True)
            st.subheader("Combined Findings (parsed)")
            st.dataframe(df_combined)

            # show category counts
            if "Category" in df_combined.columns:
                cat_count = df_combined["Category"].value_counts().reset_index()
                cat_count.columns = ["Category", "Count"]
                chart = alt.Chart(cat_count).mark_bar().encode(
                    x=alt.X("Category:N", sort='-y'),
                    y="Count:Q",
                    tooltip=["Category","Count"]
                ).properties(width=800, height=300)
                st.altair_chart(chart)

            # -----------------------
            # Post-process: Severity, Time Horizon, Priority, Business KPIs
            # -----------------------
            st.subheader("Enhance findings: Severity, Priority & KPI linkage")
            if st.button("Run Enhancement (Severity/Priority/KPIs)"):
                # pass the markdown table to the enhancer prompt
                table_md = df_combined.to_markdown(index=False)
                enhance_prompt = ENHANCE_FINDINGS_PROMPT.format(table=table_md)
                enhanced_md = call_together(enhance_prompt, max_tokens=1600)
                if enhanced_md.startswith("❌ Error"):
                    st.error(enhanced_md)
                else:
                    st.markdown(enhanced_md)
                    # parse enhanced table and show dataframe if possible
                    df_enhanced = parse_markdown_table(enhanced_md)
                    if not df_enhanced.empty:
                        st.subheader("Enhanced Findings (table view)")
                        st.dataframe(df_enhanced)

            # save combined findings to client history
            client_history.setdefault("analyses", []).append({
                "type": "document_combined",
                "ts": str(datetime.utcnow()),
                "result": df_combined.to_markdown(index=False)
            })
            save_client_history(client_name, client_history)

# -----------------------
# TAB: Incident Logs
# -----------------------
with tabs[2]:
    st.header("Incident Logs Analysis")
    incident_file = st.file_uploader("Upload Incident Log (CSV/XLSX)", type=["csv","xlsx"])
    if incident_file:
        if incident_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(incident_file)
        else:
            df_raw = pd.read_excel(incident_file)

        if df_raw.empty:
            st.error("Uploaded incident file is empty or not readable.")
        else:
            df_raw.columns = [c.strip() for c in df_raw.columns]

            def find_col(cols, patterns):
                for p in patterns:
                    for c in cols:
                        if re.search(p, c, re.IGNORECASE):
                            return c
                return None

            incident_id_col = find_col(df_raw.columns, [r"incident.?id", r"^id$", r"incident"])
            type_col = find_col(df_raw.columns, [r"\btype\b", r"category", r"incident.?type"])
            open_col = find_col(df_raw.columns, [r"open", r"start", r"reported"])
            close_col = find_col(df_raw.columns, [r"close", r"resolved", r"end"])
            desc_col = find_col(df_raw.columns, [r"description", r"desc", r"details", r"summary"])

            if not incident_id_col:
                df_raw.insert(0, "Incident ID", [f"ROW{idx+1}" for idx in range(len(df_raw))])
                incident_id_col = "Incident ID"
            if not type_col:
                df_raw["Type"] = "Unknown"
                type_col = "Type"
            if not open_col:
                df_raw["Open Date"] = pd.NaT
                open_col = "Open Date"
            if not close_col:
                df_raw["Close Date"] = pd.NaT
                close_col = "Close Date"
            if not desc_col:
                df_raw["Description"] = ""
                desc_col = "Description"

            # normalize dates
            df_raw[open_col] = pd.to_datetime(df_raw[open_col], errors='coerce')
            df_raw[close_col] = pd.to_datetime(df_raw[close_col], errors='coerce')

            st.subheader("Preview (first 10 rows)")
            st.dataframe(df_raw.head(10))

            # compute occurrences
            df_raw["__occurrences"] = df_raw.groupby(incident_id_col)[incident_id_col].transform("count")

            # Aggregate by Type for summary table
            agg = df_raw.groupby(type_col).agg(
                Incident_IDs = (incident_id_col, lambda s: ", ".join(map(str, sorted(set(s))))),
                No_of_Occurrences = (incident_id_col, lambda s: int(s.size)),
                Open_Date = (open_col, lambda s: pd.to_datetime(s, errors='coerce').min()),
                Close_Date = (close_col, lambda s: pd.to_datetime(s, errors='coerce').max()),
                Description = (desc_col, lambda s: " | ".join(map(str, list(dict.fromkeys([x for x in s if pd.notna(x) and str(x).strip()!=''])) ) ) )
            ).reset_index().rename(columns={type_col: "Type"})

            st.subheader("Aggregated Incident Summary (by Type)")
            st.dataframe(agg)

            # Charts
            st.subheader("Charts")
            # bar chart - frequency by Type
            bar = alt.Chart(agg).mark_bar().encode(
                x=alt.X("Type:N", sort='-y', title="Incident Type"),
                y=alt.Y("No_of_Occurrences:Q", title="Occurrences"),
                tooltip=["Type","No_of_Occurrences"]
            ).properties(width=800, height=300)
            st.altair_chart(bar)

            # pie chart
            pie = alt.Chart(agg).mark_arc().encode(
                theta=alt.Theta(field="No_of_Occurrences", type="quantitative"),
                color=alt.Color(field="Type", type="nominal"),
                tooltip=["Type","No_of_Occurrences"]
            ).properties(width=400, height=400)
            st.altair_chart(pie)

            # timeline: incidents opened over time (monthly)
            if df_raw[open_col].notna().any():
                ts = df_raw.set_index(open_col).resample('M').size().reset_index(name='count')
                ts['open_month'] = ts[open_col].dt.to_period('M').astype(str)
                line = alt.Chart(ts).mark_line(point=True).encode(
                    x='open_month:T',
                    y='count:Q',
                    tooltip=['open_month','count']
                ).properties(width=800, height=300)
                st.altair_chart(line)

            # Chunking incident rows and call LLM for deeper insights (in batches)
            st.subheader("LLM-based Root Cause, Risks & Recommendations")
            batch_rows = st.number_input("Batch rows per LLM call", min_value=50, max_value=2000, value=300, step=50)
            aggregated_tables = []
            table_texts = []
            for i, df_chunk in enumerate(chunk_dataframe_rows(df_raw, batch_rows)):
                # convert chunk to CSV-like text for prompt
                table_text = df_chunk[[incident_id_col, type_col, open_col, close_col, desc_col]].to_csv(index=False)
                table_texts.append(table_text)
                prompt = INCIDENT_PROMPT_TEMPLATE.format(table_text=table_text)
                st.write(f"Analyzing incident chunk {i+1}")
                resp = call_together(prompt, max_tokens=1500)
                if resp.startswith("❌ Error"):
                    st.error(resp)
                    break
                aggregated_tables.append(resp)

            # merge markdown tables returned by LLM
            full_insights_md = "\n\n".join(aggregated_tables)
            st.subheader("LLM Insights (merged)")
            st.markdown(full_insights_md)

            # store incident analysis in client history
            client_history.setdefault("analyses", []).append({
                "type": "incidents",
                "ts": str(datetime.utcnow()),
                "summary_table": agg.to_dict(orient="records"),
                "llm_insights_md": full_insights_md
            })
            save_client_history(client_name, client_history)

            # -----------------------
            # Offer correlation and gap analysis linking incidents to document findings (if available)
            # -----------------------
            st.subheader("Cross-Source Correlation & Benchmarks")
            if st.button("Run Cross-Source Correlation (Docs ↔ Incidents)"):
                # fetch last document combined analysis for this client if present
                doc_findings_md = ""
                for a in reversed(client_history.get("analyses", [])):
                    if a.get("type", "").startswith("document"):
                        doc_findings_md = a.get("result", "")
                        break
                incident_findings_md = full_insights_md
                corr_prompt = CORRELATION_PROMPT.format(doc_findings=doc_findings_md, incident_findings=incident_findings_md)
                corr_resp = call_together(corr_prompt, max_tokens=1400)
                if corr_resp.startswith("❌ Error"):
                    st.error(corr_resp)
                else:
                    st.markdown("**Correlations and consolidated recommendations:**")
                    st.markdown(corr_resp)

            if st.checkbox("Run Gap-to-Benchmark Scoring (example checks)"):
                # simple example: check for Dev:QA ratio in doc findings text
                doc_text_for_search = ""
                for a in reversed(client_history.get("analyses", [])):
                    if a.get("type", "").startswith("document"):
                        doc_text_for_search = a.get("result", "")
                        break
                if doc_text_for_search:
                    # simple regex to find something like "35:9" or "35 : 9"
                    m = re.search(r"(\d+)\s*[:]\s*(\d+)", doc_text_for_search)
                    if m:
                        num = int(m.group(1))
                        den = int(m.group(2))
                        if den != 0:
                            current_ratio = num / den
                            benchmark = BENCHMARKS["dev_to_qa"][0] / BENCHMARKS["dev_to_qa"][1]
                            deviation = (current_ratio / benchmark - 1) * 100
                            st.write(f"Detected ratio {num}:{den} -> {current_ratio:.2f}. Benchmark {BENCHMARKS['dev_to_qa'][0]}:{BENCHMARKS['dev_to_qa'][1]} -> {benchmark:.2f}. Deviation: {deviation:.1f}%")
                    else:
                        st.info("No simple dev:qa ratio pattern (e.g., '35:9') detected in last document findings.")

# -----------------------
# Sidebar: What-If Simulator (global)
# -----------------------

st.sidebar.header("What-If Simulator (Benchmark-Based)")

# Current headcounts input
current_pms = st.sidebar.number_input("Current PM Count", min_value=0, value=6)
current_bas = st.sidebar.number_input("Current BA Count", min_value=0, value=5)
current_devs = st.sidebar.number_input("Current Dev Count", min_value=0, value=35)
current_qas = st.sidebar.number_input("Current QA Count", min_value=0, value=9)

# Cost inputs (Lakh ₹ per year)
pm_cost = st.sidebar.number_input("Cost per PM (₹L)", min_value=0, value=25)
ba_cost = st.sidebar.number_input("Cost per BA (₹L)", min_value=0, value=20)
dev_cost = st.sidebar.number_input("Cost per Dev (₹L)", min_value=0, value=18)
qa_cost = st.sidebar.number_input("Cost per QA (₹L)", min_value=0, value=15)

if st.sidebar.button("Calculate Optimal Team"):
    # Get benchmark ratios
    bench_pm, bench_ba, bench_dev, bench_qa = BENCHMARKS["pm_ba_dev_qa"]

    # Use Dev as scaling base for ratio
    optimal_pms = round((current_devs / bench_dev) * bench_pm)
    optimal_bas = round((current_devs / bench_dev) * bench_ba)
    optimal_qas = round((current_devs / bench_dev) * bench_qa)

    # Calculate reductions
    pm_reduction = current_pms - optimal_pms
    ba_reduction = current_bas - optimal_bas
    qa_reduction = current_qas - optimal_qas

    # Savings
    total_savings = max(pm_reduction, 0) * pm_cost + \
                    max(ba_reduction, 0) * ba_cost + \
                    max(qa_reduction, 0) * qa_cost

    st.sidebar.markdown("### Optimal Headcount vs Current")
    st.sidebar.write(f"PMs: {current_pms} → {optimal_pms} ({pm_reduction:+})")
    st.sidebar.write(f"BAs: {current_bas} → {optimal_bas} ({ba_reduction:+})")
    st.sidebar.write(f"QAs: {current_qas} → {optimal_qas} ({qa_reduction:+})")

    st.sidebar.markdown("### Projected Savings")
    if total_savings > 0:
        st.sidebar.success(f"₹{total_savings} Lakhs/year savings possible")
    else:
        st.sidebar.info("No cost savings — team is at or below benchmark size.")

# -----------------------
# End of file
# -----------------------
