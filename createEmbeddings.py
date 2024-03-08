from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PC
from configparser import ConfigParser

import os

def readConfig():
    config = ConfigParser()
    config.read('config.ini')
    return config
def main():
    settings = readConfig()
    file_path = settings.get('settings', 'file_path')
    HUGGINGFACEHUB_API_TOKEN = settings.get('settings', 'HUGGINGFACEHUB_API_TOKEN')
    PINECONE_API_KEY = settings.get('settings', 'PINECONE_API_KEY')
    index_name = settings.get('settings', 'index_name')
    loader = UnstructuredExcelLoader(file_path)
    data = loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs=text_splitter.split_documents(data)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    docsearch=PC.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

if __name__ == "__main__":
    main()
