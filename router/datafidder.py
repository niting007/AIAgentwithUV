from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
from langchain.schema import Document

from repositories.vectordb_manager import ChromaDBManager  # Import your manager class here

router = APIRouter()

manager = ChromaDBManager()


# Request models
class AddDocsRequest(BaseModel):
    contents: List[str]
    ids: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    dates: Optional[List[str]] = None

class DeleteDocsRequest(BaseModel):
    ids: List[str]

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5


@router.post("/create")
def create_db():
    try:
        manager.create()
        return {"message": "Vector DB created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add")
def add_documents(req: AddDocsRequest):
    try:
        docs = []
        for i, content in enumerate(req.contents):
            metadata = {}
            if req.sources and len(req.sources) > i:
                metadata["source"] = req.sources[i]
            if req.dates and len(req.dates) > i:
                metadata["date"] = req.dates[i]
            doc_id = req.ids[i] if req.ids and len(req.ids) > i else None
            doc = Document(page_content=content, metadata=metadata, id=doc_id)
            docs.append(doc)

        manager.add(docs, ids=req.ids)
        return {"message": f"Added {len(docs)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete")
def delete_documents(req: DeleteDocsRequest):
    try:
        manager.delete(req.ids)
        return {"message": f"Deleted documents with ids: {req.ids}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
def reset_db():
    try:
        manager.reset()
        return {"message": "Vector DB reset (deleted)." }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/read")
def read_query(req: QueryRequest):
    try:
        results = manager.read(req.query, k=req.k)
        return {"results": [ {"content": d.page_content, "metadata": d.metadata} for d in results ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def status():
    try:
        response = manager.status()
        print("Status printed in server logs.")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
