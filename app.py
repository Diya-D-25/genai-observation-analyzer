import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import fitz  # PyMuPDF
import docx
import os

# Use API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

st.title("GenAI Observation & Risk Extractor")

uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
if uploaded_file:
    with st.spinner("Extracting text..."):
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        else:
            doc = docx.Document(uploaded_file)
            text = "\n".join(p.text for p in doc.paragraphs)

    prompt = '''You are a product assessment analyst. Given stakeholder notes or project documents, extract:
- Observation
- Associated risk
- Suggested recommendation
- Category (like Architecture, Infra, Roadmap, etc.)
Return results in a markdown table with columns: Category, Observation, Risk, Recommendation.'''

    if text:
        with st.spinner("Analyzing with GPT-4..."):
            chat = ChatOpenAI(temperature=0, model="gpt-4")
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=text[:4000])  # Trim to token limit
            ]
            result = chat(messages).content
        st.markdown(result)
    else:
        st.warning("No text extracted from document.")
