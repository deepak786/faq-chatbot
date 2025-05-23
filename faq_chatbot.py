from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st

st.title("FAQ Chatbot")

# initialize the Ollama embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Define persist directory
PERSIST_DIRECTORY = "./chroma_db/faq_chatbot"

# Try to load existing DB, create new one if it doesn't exist
if os.path.exists(PERSIST_DIRECTORY):
    with st.spinner("Loading existing Chroma database..."):
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
else:
    with st.spinner("Creating new Chroma database..."):
        # load the FAQs
        faq_file_path = "data/faqs.txt"
        loader = TextLoader(faq_file_path)
        data = loader.load()

        # split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = text_splitter.split_documents(data)
        
        # create the Chroma vector store
        db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIRECTORY)

# create the retriever
retriever = db.as_retriever()

# Initialize the Ollama chat model
model = ChatOllama(model="llama3.2") # I have llama3.2 installed on my machine
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that provides direct answers from the FAQ database.
    
    Rules:
    1. Only answer based on the provided FAQs
    2. If the exact question isn't in the FAQs, respond with "I don't have information about that in the FAQs"
    3. Keep the answer concise and to the point
    4. Do not add any general advice or suggestions not present in the FAQs
    
    Query: {query}
    FAQs: {docs}
    
    Answer:
    """
)

chain = prompt | model | parser

# Initialize session state
if "query" not in st.session_state:
    st.session_state.query = ""
    
if "history" not in st.session_state:
    st.session_state.history = []
    
def handle_submit():
    query = st.session_state.query
    
    if query:
        with st.spinner("Thinking..."):        
            # get the answer
            docs = retriever.invoke(query, k=2)
            response = chain.invoke({"docs": docs, "query": query})
            
            # clear the query input
            st.session_state.query = ""
            
            # add the question and answer to the history
            st.session_state.history.append({"question": query, "answer": response})

# Chat interface
with st.form("query_form", clear_on_submit=True):
    st.text_input("Ask a question about the FAQs", key="query")
    # Invisible button, but still triggers on Enter key
    st.form_submit_button("Submit", on_click=handle_submit)
    
# display the history
for idx, msg in enumerate(reversed(st.session_state.history)):
    with st.expander(msg['question'], expanded= True):
        st.write(msg['answer'])
