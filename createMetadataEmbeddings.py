from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from configparser import ConfigParser
from langchain_community.vectorstores import Chroma


import xlrd, mmap, os

# Define the columns we want to embed vs which ones we want in metadata
columns_to_embed = ["Product","Version","Case Number","Case Owner","Problem Statement","Original Description", "Account Number","Account Name"]
columns_to_metadata = ["Product","Version","Case Number","Case Owner","Problem Statement","Original Description", "Account Number","Account Name"]

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

def readConfig():
    config = ConfigParser()
    config.read('config.ini')
    return config

def XLSDictReader(f, sheet_index=0):
    book = xlrd.open_workbook(file_contents=mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
    sheet = book.sheet_by_index(sheet_index)
    headers = dict((i, sheet.cell_value(0, i)) for i in range(sheet.ncols))

    return (dict((headers[j], sheet.cell_value(i, j)) for j in headers) for i in range(1, sheet.nrows))

def main ():
    settings = readConfig()
    file_path = settings.get('settings', 'file_path')
    HUGGINGFACEHUB_API_TOKEN = settings.get('settings', 'HUGGINGFACEHUB_API_TOKEN')
    docs= []
    f = open(file_path)
    xls_dicts = XLSDictReader(f)
    for i, row in enumerate(xls_dicts):
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        try:
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
        except AttributeError:
            to_embed = "\n".join(f"{k}: {v}" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

    splitter = CharacterTextSplitter(separator="\n",
                                     chunk_size=500,
                                     chunk_overlap=0,
                                     length_function=len)
    documents = splitter.split_documents(docs)

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"})
    db.persist()


if __name__ == "__main__":
    main()

