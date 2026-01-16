from typing import List, Dict, Optional, Any
from sqlmodel import SQLModel, Field, JSON, Column
from pydantic import BaseModel
from datetime import datetime
import uuid

# --- Database Models (tables) ---

class Workflow(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True)
    graph_json: Dict = Field(default={}, sa_column=Column(JSON)) # Stores React Flow JSON
    execution_plan: List[str] = Field(default=[], sa_column=Column(JSON)) # Optimized list of Node IDs
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Document(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    filename: str
    file_hash: str = Field(index=True, unique=True) # MD5 hash
    storage_path: str
    vector_collection_id: str # ChromaDB collection ID
    created_at: datetime = Field(default_factory=datetime.utcnow)

# --- Pydantic Data Models (for API) ---

class NodeType(str): # Simple string alias for now, or maintain Enum
    INPUT = "inputNode"
    KNOWLEDGE_BASE = "knowledgeNode"
    LLM_ENGINE = "llmNode"
    OUTPUT = "outputNode"

class NodeData(BaseModel):
    label: str
    config: Dict[str, Any] = {}

class Node(BaseModel):
    id: str
    type: str
    data: NodeData
    position: Dict[str, float] = {"x": 0, "y": 0}

class Edge(BaseModel):
    id: str
    source: str
    target: str

class WorkflowRequest(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    user_query: str
    workflow_id: Optional[str] = None # For running saved workflows
