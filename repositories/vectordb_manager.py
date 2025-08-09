import os
from rich import print
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()  # Load .env file automatically

class ChromaDBManager:
    def __init__(self):
        self.csv_path = os.getenv("CSV_PATH", "data/entity_seed_data.csv")
        self.db_location = os.getenv("DB_LOCATION", "./chroma_langchain_db")
        self.collection_name = os.getenv("COLLECTION_NAME", "entity_collection")
        embedding_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.vector_store = None

    def create(self):
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
            if not os.path.exists(self.db_location):
                os.makedirs(self.db_location, exist_ok=True)
            df = pd.read_csv(self.csv_path)
            documents = []
            ids = []
            for _, row in df.iterrows():
                doc = Document(
                    page_content=f"{row['Name']} ({row['EntityType']}): {row['Description']}",
                    metadata={
                        "source": row["Source"],
                        "date": row["Date"],
                        "status": "unverified"
                    },
                    id=str(row["ID"])
                )
                documents.append(doc)
                ids.append(str(row["ID"]))
            
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=self.db_location
            )
            self.vector_store.add_documents(documents, ids=ids)
            print(f"Created vector store with {len(documents)} documents.")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def load(self):
        if not os.path.exists(self.db_location):
            print("DB not found. Please create first.")
            return
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.db_location
        )
        print("Vector store loaded.")

    def add(self, docs, ids=None):
        if not self.vector_store:
            self.load()
        self.vector_store.add_documents(docs, ids=ids)
        self.vector_store.persist()
        print(f"Added {len(docs)} documents.")

    def delete(self, ids):
        if not self.vector_store:
            self.load()
        self.vector_store.delete(ids=ids)
        self.vector_store.persist()
        print(f"Deleted documents with ids: {ids}")

    def reset(self):
        import shutil
        if os.path.exists(self.db_location):
            shutil.rmtree(self.db_location)
            print("Reset: Vector DB deleted.")
        else:
            print("No DB to reset.")
        self.vector_store = None

    def read(self, query, k=5):
        if not self.vector_store:
            self.load()
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        results = retriever.get_relevant_documents(query)
        return results

    def status(self):
        if not self.vector_store:
            self.load()
        try:
            count = self.vector_store._collection.count()
            print(f"Vector store document count: {count}")
            return {"status": "ok", "document_count": count}
        except Exception:
            print("Could not fetch document count.")
