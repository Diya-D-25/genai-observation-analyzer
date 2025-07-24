import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import fitz  # PyMuPDF
import docx

st.title("GenAI Observation & Risk Extractor")

# Step 1: Get API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OpenAI API key not found. Please add it under Streamlit Secrets.")
    st.stop()

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

# Step 4: Analyze with GPT
def analyze_text(text):
    system_prompt = '''You are a product assessment analyst. Given stakeholder notes or project documents, extract:
- Observation
- Associated risk
- Suggested recommendation
- Category (like Architecture, Infra, Roadmap, etc.)
Return results in a markdown table with columns: Category, Observation, Risk, Recommendation.'''

    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model="gpt-4")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text[:4000])  # Trim to fit token limit
    ]
    return chat(messages).content

# Step 5: Run the analysis
if uploaded_file:
    with st.spinner("Extracting and analyzing document..."):
        extracted_text = extract_text(uploaded_file)
        if not extracted_text.strip():
            st.error("❌ No readable text found in the document.")
        else:
            response = analyze_text(extracted_text)
            st.markdown(response)
