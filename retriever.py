# retriever.py
import ssl
import urllib3

# Parche definitivo de red corporativa para urllib3 y Pinecone
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
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Error crítico: Falta la variable OPENAI_API_KEY en el entorno.")
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("❌ Error crítico: Falta la variable PINECONE_API_KEY en el entorno.")


class CharcuteriaHybridRetriever(BaseRetriever):
    """Retriever híbrido personalizado que combina búsqueda vectorial (Pinecone) y léxica (BM25)."""
    vector_retriever: object
    bm25_retriever: object
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs_vector = self.vector_retriever.invoke(query)
        docs_bm25 = self.bm25_retriever.invoke(query)

        seen = set()
        combined = []
        for doc in docs_vector + docs_bm25:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)

        return combined[:self.k]


class CharcuteriaRAGSystem:
    """Sistema RAG Híbrido con soporte multi-formato (MD, TXT, PDF, Excel)."""

    def __init__(self, data_path: str = ".", k: int = 5):
        self.index_name = os.getenv("INDEX_NAME", "charcuteria-index")
        self.k = k

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name, embedding=self.embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )

        corpus_docs = []
        loaders = {
            ".md": (TextLoader, {"encoding": "utf-8"}),
            ".txt": (TextLoader, {"encoding": "utf-8"}),
            ".pdf": (PyPDFLoader, {}),
            ".xlsx": (UnstructuredExcelLoader, {}),
            ".xls": (UnstructuredExcelLoader, {}),
        }

        if os.path.isdir(data_path) or os.path.isfile(data_path):
            target_files = []
            if os.path.isfile(data_path):
                target_files = [data_path]
            else:
                for root, _, files in os.walk(data_path):
                    if ".venv" in root or ".git" in root or "venv" in root:
                        continue
                    for file in files:
                        if os.path.splitext(file)[1].lower() in loaders:
                            target_files.append(os.path.join(root, file))

            for file_path in target_files:
                ext = os.path.splitext(file_path)[1].lower()
                loader_cls, loader_kwargs = loaders[ext]
                try:
                    loader = loader_cls(file_path, **loader_kwargs)
                    corpus_docs.extend(loader.load())
                except Exception:
                    pass

        if not corpus_docs:
            corpus_docs = [
                Document(page_content="Documentación técnica y recetas de respaldo de Charcutería La Merced.")
            ]

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_corpus = text_splitter.split_documents(corpus_docs)

        self.bm25_retriever = BM25Retriever.from_documents(split_corpus)
        self.bm25_retriever.k = self.k

        self.hybrid_retriever = CharcuteriaHybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            k=self.k
        )

    def get_retriever(self, k: int | None = None):
        if k:
            self.k = k
            self.vector_retriever.search_kwargs = {"k": k}
            self.bm25_retriever.k = k
            self.hybrid_retriever.k = k
        return self.hybrid_retriever