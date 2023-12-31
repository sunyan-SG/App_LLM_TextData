import streamlit as st
from pathlib import Path

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import datetime
import csv
import io

st.markdown(
    """
    <div style="position: relative; width: 60px; height: 60px;">
        <div style="position: absolute; top: 16px; right: 16px;">
        <img src="https://marcusgohmarcusgoh.com/wp/wp-content/uploads/2016/02/NHG-Logo_Enhanced_jpg.jpg" alt="Image" width="100">
    </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.title("Text data analysis using GPT-4 (HSOR)")

if "openai_key" not in st.session_state:
    with st.form("API key"):
        key = st.text_input("OpenAI Key", value="", type="password")
        os.environ["OPENAI_API_KEY"] = key
        if st.form_submit_button("Submit"):
            st.session_state.openai_key = key
            st.session_state.prompt_history = []
            st.session_state.review_results = []
            st.session_state.df = None
            #st.session_state.filenames = []
            st.session_state.time = datetime.datetime.now()
            st.session_state.uploaded_files = []
            #st.session_state.result_file = "d:/result_review.csv"

if "openai_key" in st.session_state:
    if st.session_state.df is None:
        st.session_state.uploaded_files = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

    with st.form("Question"):
        question = st.text_input("Prompt:", value="", type="default")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.prompt_history.append(question)
            st.session_state.time = datetime.datetime.now()
            with st.spinner():
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                articles = []
                for file in st.session_state.uploaded_files:
                    if file.name.endswith(".pdf"):
                        #pdf_file_path = os.path.join(file_path, file)
                        pdf_data = file.read()  # Read the binary data of the PDF file
                        article = PdfReader(io.BytesIO(pdf_data))
                        articles.append(article)

                        # read text from pdf
                        raw_text = ''
                        for page in article.pages:
                            content = page.extract_text()
                            if content:
                                raw_text += content

                        # We need to split the text using Character Text Split such that it should not increse token size
                        text_splitter = CharacterTextSplitter(
                            separator="\n",
                            chunk_size=800,
                            chunk_overlap=200,
                            length_function=len,
                        )
                        texts = text_splitter.split_text(raw_text)

                        # Download embeddings from OpenAI

                        embeddings = OpenAIEmbeddings()
                        document_search = FAISS.from_texts(texts, embeddings)

                        docs = document_search.similarity_search(question)
                        result = chain.run(input_documents=docs, question=question)
                        st.write(result)
                        st.session_state.review_results.append((file.name, result))

    st.subheader("Response to the query:")
    current_time2 = datetime.datetime.now()
    time_review = current_time2-st.session_state.time

    st.write("It took GPT-4", time_review, "(hour\:min\:second)", "to response to your query")
    st.write(st.session_state.review_results)

    if st.button("Clear result history"):
        st.session_state.review_results = []

    st.subheader("Prompt history:")
    st.write(st.session_state.prompt_history)

    if st.button("Clear prompt history"):
        st.session_state.prompt_history = []


    ## Open the CSV file in write mode and write the data
    #with open(st.session_state.result_file, "w", newline="") as csv_file:
    #    csv_writer = csv.writer(csv_file)
    #    # Write the header row (optional)
    #    csv_writer.writerow(["PDF File Name", "Review Result"])
    #    # Write the data rows
    #    for row in st.session_state.review_results:
    #        csv_writer.writerow(row)

    #st.subheader("Download the AI reviewed results")
    #if st.button("Click to Download"):
    #   st.markdown(f"[Click to download the results and find it in] ({st.session_state.result_file})", unsafe_allow_html=True)

