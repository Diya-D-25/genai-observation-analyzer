import streamlit as st
import openai
import fitz  # PyMuPDF
import docx

st.title("GenAI Observation & Risk Extractor")

# Step 1: Get API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OpenAI API key not found. Please add it under Streamlit Secrets.")
    st.stop()

openai.api_key = openai_api_key

# Step 2: Upload file
uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

# Step 3: Extract text from file
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif file.name.lower().endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

# Step 4: Analyze with OpenAI directly
def analyze_text(text):
    prompt = f"""
You are a product assessment analyst. Given stakeholder notes or project documents, extract:

- Observation  
- Associated risk  
- Suggested recommendation  
- Category (like Architecture, Infra, Roadmap, etc.)

Return it in markdown table format with columns: Category, Observation, Risk, Recommendation.

Text to analyze:
{text[:4000]}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# Step 5: Run analysis
if uploaded_file:
    with st.spinner("Extracting and analyzing document..."):
        extracted_text = extract_text(uploaded_file)
        if not extracted_text.strip():
            st.error("❌ No readable text found in the document.")
        else:
            result = analyze_text(extracted_text)
            st.markdown(result)
