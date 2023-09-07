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


st.title("Literature Review using GPT-4 (HSOR)")

if "openai_key" not in st.session_state:
    with st.form("API key"):
        key = st.text_input("OpenAI Key", value="", type="password")
        os.environ["OPENAI_API_KEY"] = key
        if st.form_submit_button("Submit"):
            st.session_state.openai_key = key
            st.session_state.prompt_history = []
            st.session_state.review_results = []
            st.session_state.fp = None
            st.session_state.filenames = []
            st.session_state.time = datetime.datetime.now()
            st.session_state.result_file = []

if "openai_key" in st.session_state:
    if st.session_state.fp is None:
        fp = st.text_input(
            "Provide the path of pdf file/files for review"
            #type=["pdf", "ps"],
        )
        if fp is not None:
            file_path = Path(fp)
            st.session_state.result_file = os.path.join(file_path, "review_results.csv")
            file_names = os.listdir(file_path)
            filenames = [file for file in file_names if os.path.isfile(os.path.join(file_path, file))]

    with st.form("Question"):
        question = st.text_input("Question", value="", type="default")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.prompt_history.append(question)
            st.session_state.time = datetime.datetime.now()
            with st.spinner():
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                articles = []
                for file in filenames:
                    # articles.append(PdfReader(file_path + '\' + filenames[i]))
                    if file.endswith(".pdf"):
                        #pdf_file_path = os.path.join(str(file_path), filenames[i])  # Convert file_path to a string
                        pdf_file_path = os.path.join(file_path, file)
                        article = PdfReader(pdf_file_path)
                        articles.append(article)

                        # read text from pdf
                        raw_text = ''
                        #for page in enumerate(article.pages):
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
                        st.session_state.review_results.append((file, result))

    st.subheader("Review results of this query:")
    current_time2 = datetime.datetime.now()
    time_review = current_time2-st.session_state.time

    st.write("The time taken by ChatGPT for reviewing these PDF files is:", time_review, "seconds")
    st.write(st.session_state.review_results)

    st.subheader("Prompt history:")
    st.write(st.session_state.prompt_history)

    if st.button("Clear prompt history"):
        st.session_state.prompt_history = []
        st.session_state.fp = None

    # Open the CSV file in write mode and write the data
    with open(st.session_state.result_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row (optional)
        csv_writer.writerow(["PDF File Name", "Review Result"])
        # Write the data rows
        for row in st.session_state.review_results:
            csv_writer.writerow(row)

    st.subheader("Download the AI reviewed results")
    if st.button("Click to Download"):
        #st.download_button(label = "Download AI-Reviewed Results", data = st.session_state.result_file)
        st.markdown(f"[Click to download the results and find it in your provided path] ({st.session_state.result_file})", unsafe_allow_html=True)