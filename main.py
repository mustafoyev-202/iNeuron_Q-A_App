import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

st.title("iNeuron Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain({"query": question})

    st.header("Answer")
    st.write(response["result"])
