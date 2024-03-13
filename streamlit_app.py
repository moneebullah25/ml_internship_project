import streamlit as st
import torch
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

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

def build_qa_chain():
    torch.cuda.empty_cache()
    # Using databricks/dolly because mine OpenAI Credits are expired
    model_name = "databricks/dolly-v2-3b" # can use dolly-v2-3b or dolly-v2-7b for smaller model and faster inferences.
    # Increase max_new_tokens for a longer response, since I am on my pc I will be using only 256
    # Other settings might give better results! Play around
    instruct_pipeline = pipeline(model=model_name, device_map="cpu", trust_remote_code=True)
    prompt = PromptTemplate(input_variables=['context', 'question'], template=prompt_template)

    hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
    return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True)

def answer_question(question, Vector_db):
    similar_docs = get_similar_docs(question, Vector_db, similar_doc_count=4)
    result = qa_chain({"input_documents": similar_docs, "question": question})

    if result is None:
        return

    return result['output_text']

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction:
  Use only information in the following paragraphs to answer the question at the end.
  Explain the answer with reference to these paragraphs.
  If you don't have the information in below paragraphs then give response "I will get back to you on this".

  {context}
  
  Question: {question}

  Response:
  """

# Setup the app title
st.title('Ask AI')

# Interface for uploading documents
pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
Vector_db = None
if pdf_files:
    # Process uploaded documents
    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        all_documents.extend(documents)
    
    # Split documents and create vector store
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    all_documents = text_splitter.split_documents(all_documents)
    hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    Vector_db = Chroma.from_documents(collection_name="document_docs", documents=all_documents, embedding=hf_embed, persist_directory="Persist_dir")
    Vector_db.persist()

# Initialize question-answering pipeline
qa_chain = build_qa_chain()

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
    answer = answer_question(prompt, Vector_db)
    st.chat_message('assistant').markdown(answer)
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': answer
    })
