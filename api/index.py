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
        # Simple file save without DB dependency
        content = await file.read()
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        
        upload_dir = "uploads"
        import os
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "filename": filename,
            "vector_collection_id": file_id,
            "message": "Document uploaded successfully"
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
