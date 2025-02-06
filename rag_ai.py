import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Page config
st.set_page_config(page_title="ResearchMind AI", page_icon="🔬")

# UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #1E1E2E 0%, #2D2D44 100%);
        color: #E0E0E0;
    }
    h1, h2, h3 {
        color: #7EB2FF !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stChatMessage {
        background: rgba(30, 30, 46, 0.6) !important;
        border: 1px solid #7EB2FF !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    .stFileUploader {
        background: rgba(30, 30, 46, 0.6);
        border: 2px dashed #7EB2FF;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatInput input {
        border: 2px solid #7EB2FF !important;
        background: rgba(30, 30, 46, 0.6) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Improved prompt template
PROMPT_TEMPLATE = """
You are a precise research assistant. Follow these rules strictly:
1. Only answer based on the provided context
2. If the answer isn't in the context, say "I cannot find information about this in the document"
3. Match keywords exactly from the query to the context
4. Keep responses focused and specific

Query: {user_query} 
Context: {document_context} 
Answer: """

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query, k=4)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# UI
st.title("🔬 ResearchMind AI")
st.markdown("### Your Intelligent Research Assistant")
st.markdown("---")

uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis"
)

if uploaded_pdf:
    with st.spinner("Processing document..."):
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully!")
    
    user_input = st.chat_input("Ask your question...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🔬"):
            st.write(ai_response)