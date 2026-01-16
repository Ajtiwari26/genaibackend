from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from typing import List
import sys
import os
import uuid
from schema import WorkflowRequest, Document, Workflow
from workflow_engine import WorkflowEngine
from database import create_db_and_tables, get_db
from services import IngestionService
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = FastAPI(title="GenAI Stack Backend")

# CORS - Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    try:
        create_db_and_tables()
    except:
        pass  # Database might not be available on Vercel

@app.get("/")
def read_root():
    return {"message": "GenAI Stack API is running"}

@app.post("/workflows/validate")
def validate_workflow(request: WorkflowRequest):
    try:
        engine = WorkflowEngine(request)
        engine.validate_graph()
        return {"valid": True, "plan": engine.get_execution_plan()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_workflow")
async def run_workflow(request: WorkflowRequest):
    try:
        engine = WorkflowEngine(request)
        engine.validate_graph()
        return StreamingResponse(engine.stream_execution(), media_type="text/event-stream")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Read file content into memory (Vercel has read-only filesystem)
        content = await file.read()
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        
        # Extract text content (handle both text and binary files)
        try:
            if file.filename.endswith('.pdf'):
                # For PDFs, try to extract text (pdfplumber would be ideal but just get raw content)
                import pdfplumber
                from io import BytesIO
                text_content = ""
                try:
                    pdf_file = BytesIO(content)
                    with pdfplumber.open(pdf_file) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_content += page_text + "\n"
                except:
                    text_content = content.decode('utf-8', errors='ignore')
            else:
                # For text files
                text_content = content.decode('utf-8', errors='ignore')
        except Exception as e:
            text_content = f"File: {file.filename}"
        
        # Store the document in the mock chroma collection
        if text_content and text_content.strip():
            from services import chroma_client, embedding_model
            
            # Create collection with file_id as the collection name
            collection_name = f"col_{file_id}"
            collection = chroma_client.get_or_create_collection(name=collection_name)
            
            # Generate embedding for the document
            embeddings = embedding_model.encode([text_content])
            
            # Store in collection
            collection.add(
                ids=["doc_0"],
                documents=[text_content],
                embeddings=embeddings,
                metadatas=[{"filename": file.filename, "type": "uploaded_document"}]
            )
        
        return {
            "filename": filename,
            "vector_collection_id": file_id,
            "message": "Document uploaded and processed successfully",
            "size": len(content),
            "content_type": file.content_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    try:
        documents = db.exec(select(Document)).all()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
