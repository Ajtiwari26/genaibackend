import asyncio
import httpx
from schema import WorkflowRequest, Node, Edge, NodeType, NodeData
from workflow_engine import WorkflowEngine

async def test_engine_streaming():
    # 1. Create Mock Nodes (Same connected graph)
    node1 = Node(id="1", type=NodeType.INPUT, data=NodeData(label="User Query"), position={"x":0,"y":0})
    node2 = Node(id="2", type=NodeType.KNOWLEDGE_BASE, data=NodeData(label="PDF Loader"), position={"x":100,"y":0})
    node3 = Node(id="3", type=NodeType.LLM_ENGINE, data=NodeData(label="GPT-4", config={"system_prompt": "Be concise."}), position={"x":200,"y":0})
    node4 = Node(id="4", type=NodeType.OUTPUT, data=NodeData(label="Final Answer"), position={"x":300,"y":0})

    # 2. Create Mock Edges (Linear Flow)
    edges = [
        Edge(id="e1", source="1", target="2"),
        Edge(id="e2", source="2", target="3"),
        Edge(id="e3", source="3", target="4")
    ]

    request = WorkflowRequest(
        nodes=[node1, node2, node3, node4],
        edges=edges,
        user_query="How does this engine work?"
    )

    print("\n--- Testing Direct Engine Generator ---")
    engine = WorkflowEngine(request)
    async for chunk in engine.execute_stream():
        print(f"[STREAM] {chunk.strip()}")

async def test_validation_cycle():
    # Create a cyclic graph: 1->2->1
    node1 = Node(id="1", type=NodeType.INPUT, data=NodeData(label="A"), position={"x":0,"y":0})
    node2 = Node(id="2", type=NodeType.LLM_ENGINE, data=NodeData(label="B"), position={"x":100,"y":0})
    edges = [
        Edge(id="e1", source="1", target="2"),
        Edge(id="e2", source="2", target="1") 
    ]
    request = WorkflowRequest(nodes=[node1, node2], edges=edges, user_query="fail")
    
    engine = WorkflowEngine(request)
    print("\n--- Testing Cycle Detection ---")
    try:
        engine.validate_graph()
        print("FAILED: Use cycle was not detected.")
    except ValueError as e:
        print(f"SUCCESS: Caught expected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_engine_streaming())
    asyncio.run(test_validation_cycle())
