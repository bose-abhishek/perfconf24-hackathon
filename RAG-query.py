import requests
import json
import urllib3
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFaceEndpoint
#from langchain.chains.question_answering import load_qa_chain


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Disable SSL verification
urllib3.disable_warnings()

CHROMA_PATH = "chroma"
URL="https://llama-2-7b-chat-perfconf-hackathon.apps.dripberg-dgx2.rdu3.labs.perfscale.redhat.com"
endpoint="/generate"

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

#query = "list all Ansible performance related cases"
query = input("Enter your query to search: ")

results = db.similarity_search_with_relevance_scores(query, k=15)
#print(results)

context_text = "\n---\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt_RAG = prompt_template.format(context=context_text, question=query)
prompt = format_llama_prompt(prompt_RAG)


llm = HuggingFaceEndpoint(
    endpoint_url=f"{URL}",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.9,
    repetition_penalty=1.03,
)

#llm(query)

response = llm.predict(prompt)
print(response)

#chain = load_qa_chain(llm,chain_type="stuff")
#response = chain.run(input_documents=results, question=query)
#print(response)
