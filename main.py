from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from typing import List

from schema import WorkflowRequest, Document, Workflow
from workflow_engine import WorkflowEngine
from database import create_db_and_tables, get_db
from database import create_db_and_tables, get_db
from services import IngestionService
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = FastAPI(title="ModulusSellMobile Backend")

# CORS
origins = [
    "http://localhost:3000", 
    "http://localhost:8000", 
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "https://genai-khaki.vercel.app",
    "https://genaibackend-api.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
def read_root():
    return {"message": "Workflow Builder API is running"}

@app.post("/workflows/validate")
def validate_workflow(request: WorkflowRequest):
    try:
        engine = WorkflowEngine(request)
        engine.validate_graph()
        return {"valid": True, "plan": engine.get_execution_plan()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/run_workflow")
async def run_workflow(request: WorkflowRequest):
    """
    Executes the workflow graph and returns a StreamingResponse (SSE).
    """
    engine = WorkflowEngine(request)
    
    return StreamingResponse(
        engine.execute_stream(), 
        media_type="text/event-stream"
    )

@app.post("/documents/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    service = IngestionService(db)
    return await service.ingest_document(file)

@app.get("/documents", response_model=List[Document])
def list_documents(db: Session = Depends(get_db)):
    return db.exec(select(Document)).all()
