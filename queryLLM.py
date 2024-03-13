from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PC
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from configparser import ConfigParser
import streamlit as st
import os

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def format_llama_prompt(user_prompt):
    prompt = """\
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.  Your answers should not include any harmful, offensive, dangerous, or illegal content.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please ask for more details.
<</SYS>>

{user_prompt}[/INST]\
"""
    return prompt.format(user_prompt=user_prompt)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def readConfig():
    config = ConfigParser()
    config.read('config.ini')
    return config

def main():
    settings = readConfig()
    HUGGINGFACEHUB_API_TOKEN = settings.get('settings', 'HUGGINGFACEHUB_API_TOKEN')
    PINECONE_API_KEY = settings.get('settings', 'PINECONE_API_KEY')
    index_name = settings.get('settings', 'index_name')
    INFERENCE_URL = settings.get('settings', 'INFERENCE_URL')
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image('llama_drama.png')
    query = st.text_input("prompt", value="", key="prompt")
    if query!= "":
        #query = input("Enter your query to search: ")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
        os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        docsearch=PC.from_existing_index(index_name, embeddings)
        docs=docsearch.similarity_search(query, k=10)

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt_RAG = prompt_template.format(context=docs, question=query)

        prompt = format_llama_prompt(prompt_RAG)

        llm = HuggingFaceEndpoint(
            endpoint_url=f"{INFERENCE_URL}",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.9,
            repetition_penalty=1.03,
        )
        response = llm.invoke(prompt)
        #print(response)
        st.markdown(response)


if __name__ == "__main__":
    main()
