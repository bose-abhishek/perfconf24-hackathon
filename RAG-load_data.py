import requests
import json
import PyPDF2
import urllib3
#from chromadb import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
import shutil

# Disable SSL verification
urllib3.disable_warnings()

CHROMA_PATH = "chroma"
DATA_PATH = "/home/test/salesforce_report1.csv"

#load doc from single PDF
#loader = PyPDFLoader(DATA_PATH)
#docs = loader.load()
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

#load from csv file
loader = CSVLoader(DATA_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(separators = "\n", chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(docs)
print(len(chunks))

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

db = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"})
db.persist()
