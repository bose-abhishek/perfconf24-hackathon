import requests
import json
import PyPDF2
import urllib3
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
import shutil
import sys
from datetime import datetime

# Disable SSL verification
urllib3.disable_warnings()

CHROMA_PATH = "chroma"
DATA_PATH = "/home/test/Downloads/report1709795410593.csv"
now = datetime.now()
current_time = now.strftime("%H:%M:%S")

print("Current Time =", current_time)
#load doc from single PDFpython
#loader = PyPDFLoader(DATA_PATH)
#docs = loader.load()

#load from csv file
loader = CSVLoader(DATA_PATH)
docs = loader.load()

print(current_time, " doc loading completed")

text_splitter = RecursiveCharacterTextSplitter(separators = "\n", chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print("Number of chunks:", len(chunks))

print(current_time, " Transformer Embedding begins")
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print(current_time, " Transformer Embedding Ends")

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
print(current_time, " Adding vectors to DB begins")
db = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"})
db.persist()
print(current_time, " Adding vectors to DB Ends")
