from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from enum import Enum

class NodeType(str, Enum):
    INPUT = "inputNode"
    KNOWLEDGE_BASE = "knowledgeNode"
    LLM_ENGINE = "llmNode"
    OUTPUT = "outputNode"

# --- Data Models (Matches React Flow Structure) ---
class NodeData(BaseModel):
    label: str
    config: Dict[str, Any] = {} # Stores things like "model_name", "pdf_id", "prompt"

class Node(BaseModel):
    id: str
    type: str  # Matches NodeType values
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
