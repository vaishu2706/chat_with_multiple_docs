import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

# OPENAI imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------- Load Environment Variables ----------
load_dotenv()
os.getenv("OPENAI_API_KEY")  # Make sure .env contains OPENAI_API_KEY

INDEX_DIR = "faiss_index"


# --------- Extract PDF Text ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# --------- Text Splitting ----------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


# --------- Vector Store ----------
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(INDEX_DIR)


# --------- Chat Model / QA Chain ----------
def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant for question-answering over PDF documents.

    Use the following context to answer the question.
    - Answer as detailed as possible using ONLY the provided context.
    - If the answer is not available in the context, reply exactly:
      "answer is not available in the context"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )

    # LCEL: prompt -> LLM -> output parser (string)
    chain = prompt | llm | StrOutputParser()
    return chain


# --------- User Query ----------
def user_input(user_question):
    # 1. Check if index exists
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    if not os.path.exists(index_path):
        st.error("No index found. Please upload PDF(s) and click 'Submit & Process' first.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Load FAISS index
    new_db = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # required in new langchain-community
    )

    # 3. Retrieve similar chunks
    docs = new_db.similarity_search(user_question, k=4)

    # 4. Build context string from docs
    context = "\n\n".join(doc.page_content for doc in docs)

    # 5. Run chain
    chain = get_conversational_chain()
    response = chain.invoke(
        {
            "context": context,
            "question": user_question,
        }
    )

    print(response)
    st.write("Reply:", response)


# --------- Streamlit UI ----------
def main():
    st.set_page_config("Chat PDF")
    st.header("📄 Chat with PDF using GPT-4o-mini")

    user_question = st.text_input("Ask a question from the PDF")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF before processing.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Completed ✔")


if __name__ == "__main__":
    main()
