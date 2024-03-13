import streamlit as st
import os
import shutil
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import re

_ : bool = load_dotenv(find_dotenv()) # read local .env file

CHROMA_PATH = "Persist_dir"

def detect_sensitive_information(text):
    sensitive_info = []

    # Regular expressions for detecting sensitive information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
    credit_card_pattern = r'\b(?:\d[ -]*?){13,16}\b'

    # Detect email addresses
    emails = re.findall(email_pattern, text)
    if emails:
        sensitive_info.extend(emails)

    # Detect phone numbers
    phones = re.findall(phone_pattern, text)
    if phones:
        sensitive_info.extend(phones)

    # Detect credit card numbers
    credit_cards = re.findall(credit_card_pattern, text)
    if credit_cards:
        sensitive_info.extend(credit_cards)

    return sensitive_info

def contains_inappropriate_content(text):
    # List of censored words
    censored_words = ["fuck", "shit", "damn", "ass", "bitch", "bastard", "nigga"]

    # Check if any censored word is present in the text
    for word in censored_words:
        if word in text.split():  # Convert both text and word to lowercase for case-insensitive comparison
            return True

    return False

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def get_similar_docs(question, Vector_db, similar_doc_count):
    similar_docs = Vector_db.similarity_search(question, similar_doc_count)

    unique_docs = []
    seen_doc_contents = set()

    for doc in similar_docs:
        doc_content = doc.page_content  # Assuming 'page_content' is the key for the document content
        if doc_content not in seen_doc_contents:
            unique_docs.append(doc)
            seen_doc_contents.add(doc_content)

    return unique_docs

# Setup the app title
st.title('Ask AI')

# Interface for uploading documents
pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

# pdf_files = ["attention-is-all-you-need-Paper.pdf"]
Vector_db = None
if pdf_files:
    # Process uploaded documents
    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        all_documents.extend(documents)
    
    chunks = split_text(all_documents)
    Vector_db = Chroma.from_documents(collection_name="document_docs", 
                                      documents=chunks, 
                                      embedding=OpenAIEmbeddings(model="text-embedding-3-large"), 
                                      persist_directory=CHROMA_PATH)
    Vector_db.persist()



PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction:
  Use only information in the following paragraphs to answer the question at the end.
  Explain the answer with reference to these paragraphs.
  If you don't have the information in below paragraphs then give response "I will get back to you on this".

  {context}
  
  Question: {question}

  Response:
  """

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages |
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt here')
# If the user hits enter then

if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content':prompt})
    
    # Check if the prompt contains inappropriate content
    if contains_inappropriate_content(prompt):
        st.warning("Your prompt contains inappropriate content. Please rephrase.")
    elif detect_sensitive_information(prompt):
        st.warning("Your prompt contains sensitive information. Please rewrite.")
    else:
        # Search the DB.
        results = Vector_db.similarity_search_with_relevance_scores(prompt, k=3)
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        final_prompt = prompt_template.format(context=context_text, question=prompt)

        model = ChatOpenAI(model="gpt-3.5-turbo-1106")
        answer = model.predict(prompt)

        # Check if the generated answer contains inappropriate content
        if contains_inappropriate_content(answer):
            st.warning("The generated answer contains inappropriate content. Please try again.")
        elif detect_sensitive_information(prompt):
            st.warning("Your prompt contains sensitive information. Please rewrite.")
        else:
            st.chat_message('assistant').markdown(answer)
            st.session_state.messages.append({
                'role': 'assistant', 
                'content': answer
            })