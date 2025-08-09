import streamlit as st
import requests
import fitz  # PyMuPDF
import docx
import json
import re
import pandas as pd
import altair as alt
import io
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Phase1 — Client-aware GenAI Observation Analyzer")

# -------------------------
# Config / canonical list
# -------------------------
CANONICAL_CATEGORIES = [
    "Product Roadmap and Overview",
    "Product Portfolio and Backlog",
    "Operating Model",
    "Architecture & Tech Stack",
    "SDLC & SMART Practices",
    "Cost Management",
    "Cloud Infrastructure",
    "Governance & Decision-Making",
    "Go-to-Market & Channel Strategy",
    "Data & Analytics"
]

TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY not found in Streamlit Secrets.")
    st.stop()

# -------------------------
# File upload / helpers
# -------------------------
uploaded_file = st.file_uploader("Upload client report (PDF / DOCX / PPTX text will work best)", type=["pdf", "docx", "pptx"])

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".pptx"):
        # basic PPTX text extraction without python-pptx dependency (if present, you can replace)
        try:
            from pptx import Presentation
            file.seek(0)
            prs = Presentation(file)
            text = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
            return "\n".join(text)
        except Exception:
            return ""
    return ""

def chunk_text(text, chunk_size=1800, overlap=300):
    i = 0
    chunks = []
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def parse_json_from_text(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\[.*\]', text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return []
        return []

# -------------------------
# App discovery (extract candidate product / app names)
# -------------------------
st.sidebar.header("Client profile")
client_id = st.sidebar.text_input("Client ID (short)", value="client_x")

if uploaded_file:
    with st.spinner("Extracting raw text from document..."):
        raw_text = extract_text(uploaded_file)
    if not raw_text.strip():
        st.error("No readable text found in the uploaded file.")
    else:
        st.sidebar.success("Document loaded. Size: {:,} characters".format(len(raw_text)))

        # Discover candidate apps
        if st.sidebar.button("Discover client apps & channels"):
            st.info("Running app discovery (LLM). This may take several seconds...")
            discover_prompt = f"""
Extract unique product, application, channel or platform names from the text. For each item, return a JSON array where each object is:
{{"name":"<as-found>", "type":"app|channel|platform|team|other", "evidence":"<short snippet (max 140 chars)>" }}
Only list items that explicitly appear in the text. Return JSON only.

Text:
{raw_text[:30000]}
"""
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": "You are an assistant that extracts named products, apps and channels from client documents."},
                    {"role": "user", "content": discover_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 800
            }
            r = requests.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                resp_text = r.json()["choices"][0]["message"]["content"]
                candidates = parse_json_from_text(resp_text)
                if not candidates:
                    st.warning("No candidates discovered by LLM.")
                    candidates = []
            else:
                st.error(f"Discovery failed: {r.status_code} - {r.text}")
                candidates = []

            # Show confirmation UI for candidates
            st.subheader("Discovered candidates — confirm / canonicalize")
            if len(candidates) == 0:
                st.info("No candidates to confirm.")
            else:
                mapping = []
                for idx, c in enumerate(candidates):
                    col1, col2 = st.columns([2,3])
                    with col1:
                        st.write(f"**Candidate:** {c.get('name','')}")
                        st.write(f"*Type:* {c.get('type','')}")
                        st.write(f"*Evidence:* {c.get('evidence','')}")
                    with col2:
                        canonical = st.text_input(f"Canonical name for #{idx+1}", value=c.get('name',''), key=f"can_{idx}")
                        override = st.selectbox(f"Category override for #{idx+1} (optional)", options=[""]+CANONICAL_CATEGORIES, key=f"ov_{idx}")
                        keep = st.checkbox(f"Keep this candidate", value=True, key=f"keep_{idx}")
                        if keep:
                            mapping.append({"canonical": canonical.strip(), "aliases": [c.get('name','').strip()], "category_override": override})

                if st.button("Save client profile mapping"):
                    # save a simple JSON file to disk (in Streamlit environment)
                    profile = {
                        "client_id": client_id,
                        "canonical_apps": mapping,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    profile_path = f"client_profile_{client_id}.json"
                    with open(profile_path, "w") as f:
                        json.dump(profile, f, indent=2)
                    st.success(f"Saved client profile to {profile_path}")
                    st.write(profile)

        # Option to upload an existing client profile
        st.sidebar.markdown("---")
        uploaded_profile = st.sidebar.file_uploader("Or upload existing client_profile JSON", type=["json"])
        client_profile = None
        if uploaded_profile:
            try:
                client_profile = json.load(uploaded_profile)
                st.sidebar.success("Loaded uploaded client profile.")
            except Exception as e:
                st.sidebar.error("Invalid JSON profile.")
        else:
            # try to load saved file if exists
            try:
                profile_path = f"client_profile_{client_id}.json"
                with open(profile_path, "r") as f:
                    client_profile = json.load(f)
                st.sidebar.info(f"Loaded saved profile: {profile_path}")
            except Exception:
                client_profile = None

        # -------------------------
        # Run full analysis (chunking + JSON output)
        # -------------------------
        if st.button("Run full Phase 1 analysis"):
            if client_profile is None:
                st.warning("No client profile loaded — it's recommended to run 'Discover client apps' and save mapping first. Proceeding without mapping.")
            st.info("Chunking and running LLM for deep extraction. This may take a while depending on document size and API speed.")

            chunks = chunk_text(raw_text, chunk_size=1800, overlap=300)
            all_findings = []
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}

            # JSON prompt template that requires category field
            json_prompt_template = """
You are a senior product & technology strategy consultant. Analyze the following TEXT CHUNK and return a JSON array of findings.
Each finding must be an object with these keys:
- category: one of the canonical categories if appropriate, else a short descriptive category string.
- observation: concise, specific finding.
- risk: detailed, explaining root cause, stakeholders affected, and business/operational impact.
- recommendation: actionable, prioritized suggestion.
- apps: array of any product/application/channel names explicitly mentioned in the observation (as they appear).
Return JSON only (no explanation).
TEXT_CHUNK:
"""
            progress_bar = st.progress(0)
            for idx, ch in enumerate(chunks):
                prompt = json_prompt_template + "\n" + ch[:3500]
                payload = {
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that returns JSON findings from text."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1200
                }
                r = requests.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    resp_text = r.json()["choices"][0]["message"]["content"]
                    items = parse_json_from_text(resp_text)
                    if not isinstance(items, list):
                        items = []
                    for it in items:
                        # enrich with metadata
                        it['source_chunk_idx'] = idx
                        it['source_snippet'] = ch[:300]
                        all_findings.append(it)
                else:
                    st.error(f"LLM call failed for chunk {idx}: {r.status_code}")
                progress_bar.progress(int((idx+1)/len(chunks)*100))

            st.success(f"Extraction complete — {len(all_findings)} findings collected.")

            if len(all_findings) == 0:
                st.info("No findings returned by LLM.")
            else:
                df = pd.DataFrame(all_findings)
                # normalize missing columns
                for c in ["category","observation","risk","recommendation","apps","source_chunk_idx","source_snippet"]:
                    if c not in df.columns:
                        df[c] = ""

                # Map apps based on client_profile if present
                def map_apps(raw_apps):
                    if not isinstance(raw_apps, list):
                        return []
                    mapped = []
                    if client_profile:
                        for a in raw_apps:
                            found = False
                            for entry in client_profile.get("canonical_apps", []):
                                aliases = [x.lower() for x in entry.get("aliases",[])]
                                if a.strip().lower() in aliases or a.strip().lower() == entry.get("canonical","").strip().lower():
                                    mapped.append(entry.get("canonical"))
                                    found = True
                                    break
                            if not found:
                                mapped.append(a.strip())
                    else:
                        mapped = [a.strip() for a in raw_apps]
                    # unique preserving order
                    seen=set(); out=[]
                    for x in mapped:
                        if x not in seen:
                            seen.add(x); out.append(x)
                    return out

                df['apps_mapped'] = df['apps'].apply(map_apps)

                # Apply category_override if any mapping says so
                def apply_category_override(row):
                    cat = row.get('category','')
                    if client_profile:
                        for entry in client_profile.get("canonical_apps", []):
                            if entry.get("canonical") in (row.get('apps_mapped') or []):
                                override = entry.get("category_override","")
                                if override:
                                    return override
                    return cat
                df['category_canonical'] = df.apply(apply_category_override, axis=1)

                # Deduplicate similar observations (simple lower-trim)
                df['obs_norm'] = df['observation'].astype(str).str.strip().str.lower()
                df = df.drop_duplicates(subset=['obs_norm']).reset_index(drop=True)

                # Anonymize toggle
                anonymize = st.checkbox("Anonymize client/app names for display & export", value=False)
                df_display = df.copy()
                if anonymize:
                    # simple anonymization: replace canonical app names with ClientX_App1 ... ClientX_AppN
                    apps_list = sorted({a for lst in df_display['apps_mapped'] for a in lst})
                    mapping_app = {a: f"ClientApp{idx+1}" for idx,a in enumerate(apps_list)}
                    def anonymize_apps(l):
                        return [mapping_app.get(x, x) for x in l]
                    df_display['apps_mapped'] = df_display['apps_mapped'].apply(anonymize_apps)
                    # also replace client name occurrences in text fields
                    for col in ['observation','risk','recommendation','source_snippet']:
                        df_display[col] = df_display[col].astype(str).replace('|'.join(apps_list), lambda m: mapping_app.get(m.group(0), m.group(0)), regex=True)

                # Show basic table
                st.subheader("Findings (table)")
                st.dataframe(df_display[['category_canonical','observation','risk','recommendation','apps_mapped','source_chunk_idx']])

                # Charts: count by category & by app
                st.subheader("Charts")
                # category count
                cat_count = df_display.groupby('category_canonical').size().reset_index(name='count').sort_values('count', ascending=False)
                if not cat_count.empty:
                    chart_cat = alt.Chart(cat_count).mark_bar().encode(
                        x=alt.X('category_canonical:N', sort='-y', title='Category'),
                        y=alt.Y('count:Q', title='Count'),
                        tooltip=['category_canonical','count']
                    ).properties(width=700, height=300)
                    st.altair_chart(chart_cat)
                # apps breakdown
                rows=[]
                for _, r in df_display.iterrows():
                    for a in r['apps_mapped']:
                        rows.append({'app': a, 'category': r['category_canonical']})
                if rows:
                    apps_df = pd.DataFrame(rows)
                    apps_count = apps_df.groupby('app').size().reset_index(name='count').sort_values('count', ascending=False)
                    chart_app = alt.Chart(apps_count).mark_bar().encode(
                        x=alt.X('app:N', sort='-y', title='App'),
                        y=alt.Y('count:Q', title='Count'),
                        tooltip=['app','count']
                    ).properties(width=700, height=300)
                    st.altair_chart(chart_app)

                # Excel export
                buf = io.BytesIO()
                df_export = df_display.copy()
                df_export = df_export.drop(columns=['obs_norm'], errors='ignore')
                df_export.to_excel(buf, index=False, engine='openpyxl')
                buf.seek(0)
                st.download_button("Download findings as Excel", buf, file_name=f"findings_{client_id}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
