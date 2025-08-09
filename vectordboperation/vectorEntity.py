from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from rich import print
import pandas as pd

df = pd.read_csv("data/entity_seed_data.csv")
embedding = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./vector_db/chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=f"{row['Name']} ({row['EntityType']}): {row['Description']}",
            metadata={
                "source": row["Source"],
                "date": row["Date"],
                "status": "unverified"  # Start as unknown
            },
            id=str(row["ID"])
        )
        ids.append(str(row["ID"]))
        documents.append(document)

vector_store = Chroma(
    collection_name="entity_collection",
    embedding_function=embedding,
    persist_directory=db_location
)
if add_documents:
    vector_store.add_documents(documents, ids=ids)
    # vector_store.persist()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)