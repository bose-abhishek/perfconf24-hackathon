import requests
import json
import argparse
import urllib3
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import VectorDBQA

# Disable SSL verification
urllib3.disable_warnings()

CHROMA_PATH = "chroma"

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

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=db)

#query = "what is PG autoscaling?"
query = input("Enter your query to search: ")

results = db.similarity_search_with_relevance_scores(query, k=15)
#print(results)

context_text = "\n---\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt_RAG = prompt_template.format(context=context_text, question=query)
#print(prompt_RAG)

URL="https://llama-2-7b-chat-perfconf-hackathon.apps.dripberg-dgx2.rdu3.labs.perfscale.redhat.com"

endpoint="/generate"

headers = {
        "Content-Type": "application/json"
        }

prompt = format_llama_prompt(prompt_RAG)

data = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 600,
        "temperature": 0.9, # Just an example
        "repetition_penalty": 1.03, # Just an example
        "details": False
        }
}

response = requests.post(f"{URL}{endpoint}", headers=headers, json=data, verify=False) #, stream=True)
print(response.json().get("generated_text"))
