# ingest.py
import ssl
import urllib3

# Parche definitivo de red corporativa
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

old_init = urllib3.poolmanager.PoolManager.__init__
def new_init(self, *args, **kwargs):
    kwargs['cert_reqs'] = 'CERT_NONE'
    old_init(self, *args, **kwargs)
urllib3.poolmanager.PoolManager.__init__ = new_init

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_PATH = "."


def indexar_documentos_charcuteria():
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("INDEX_NAME", "charcuteria-index")

    if not api_key or not pinecone_api_key:
        raise ValueError("Faltan credenciales de OpenAI o Pinecone en las variables de entorno.")

    pc = Pinecone(api_key=pinecone_api_key)

    try:
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    except Exception as e:
        print(f"⚠️ Aviso de conexión: {e}")
        existing_indexes = []

    if index_name not in existing_indexes:
        print(f"☁️ Creando índice Serverless '{index_name}' en Pinecone...")
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("✅ ¡Índice creado con éxito!")
        except Exception as ex:
            print(f"ℹ️ Nota sobre el índice: {ex}")
    else:
        print(f"ℹ️ El índice '{index_name}' ya existe en la nube.")

    print("📁 Cargando base de conocimiento...")
    loaders = {
        ".md": (TextLoader, {"encoding": "utf-8"}),
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".pdf": (PyPDFLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".xls": (UnstructuredExcelLoader, {}),
    }

    docs = []
    for root, _, files in os.walk(DATA_PATH):
        if ".venv" in root or ".git" in root or "venv" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in loaders:
                file_path = os.path.join(root, file)
                loader_cls, loader_kwargs = loaders[ext]
                try:
                    print(f"   📄 Procesando: {file}")
                    loader = loader_cls(file_path, **loader_kwargs)
                    docs.extend(loader.load())
                except Exception as ex:
                    print(f"   ⚠️ No se pudo cargar {file}: {ex}")

    if not docs:
        print("❌ No se encontraron documentos compatibles.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Se generaron {len(chunks)} fragmentos.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    print("🚀 Subiendo vectores a Pinecone...")
    PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)
    print("✅ ¡Ingesta completada con éxito!")


if __name__ == "__main__":
    indexar_documentos_charcuteria()