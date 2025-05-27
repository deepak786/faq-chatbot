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
    1. If the exact question is not found in the FAQs, politely inform the user: "I don't have information about that in the FAQs," and then provide the best possible answer strictly based on related FAQ content only.
    2. Do NOT mention document IDs, metadata, or any internal references.
    3. Keep the answer concise and to the point.
    4. Do NOT add any advice, offers, or suggestions beyond what is present in the FAQs.
    
    Query: {query}
    FAQs: {docs}
    
    Answer:
    """
)

chain = prompt | model | parser

# Define the query expansion chain
query_expansion_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert in search query expansion. Your task is to enhance the given question by adding relevant synonyms in parentheses, while preserving the exact same intent and action.

    Rules:
    1. Produce a SINGLE expanded search query.
    2. Add relevant synonyms in parentheses immediately after each key word or phrase.
    3. Maintain the EXACT SAME intent and action as the original query.
    4. DO NOT introduce different actions, topics, or unrelated terms.
    5. Return ONLY the expanded query. Do not provide explanations or additional text.

    Example input: "How to change my username?"
    Example output: How to (change modify update edit revise) my (username user-name display-name account-name)

    Example input: "What are the payment methods?"
    Example output: What are the (payment billing transaction) (methods options ways means)

    Question: {query}
    """
)

expansion_chain = query_expansion_prompt | model | parser

# Initialize session state
if "query" not in st.session_state:
    st.session_state.query = ""
    
if "history" not in st.session_state:
    st.session_state.history = []
    
def handle_submit():
    query = st.session_state.query
    
    if query:
        with st.spinner("Thinking..."):
            # expand the query to retrieve more relevant FAQs
            expanded_query = expansion_chain.invoke({"query": query})
            
            # get the answer based on the expanded query
            docs = retriever.invoke(expanded_query, k=2)
            
            # generate the response based on original query but documents from expanded query
            response = chain.invoke({"docs": docs, "query": query})
            
            # clear the query input
            st.session_state.query = ""
            
            # add the question and answer to the history
            st.session_state.history.append({"question": query, "answer": response, "expanded_query": expanded_query})

# Chat interface
with st.form("query_form", clear_on_submit=True):
    st.text_input("Ask a question about the FAQs", key="query")
    # Invisible button, but still triggers on Enter key
    st.form_submit_button("Submit", on_click=handle_submit)
    
# display the history
for idx, msg in enumerate(reversed(st.session_state.history)):
    with st.expander(msg['question'], expanded= True):
        st.write(f"Expanded Query: {msg['expanded_query']}")
        st.write(f"Answer: {msg['answer']}")
