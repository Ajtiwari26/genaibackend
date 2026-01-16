import asyncio
from models import WorkflowRequest, Node, Edge, NodeType, NodeData
from workflow_engine import WorkflowEngine

async def test_engine():
    # 1. Create Mock Nodes
    node1 = Node(id="1", type=NodeType.INPUT, data=NodeData(label="User Query"))
    node2 = Node(id="2", type=NodeType.KNOWLEDGE_BASE, data=NodeData(label="PDF Loader"))
    node3 = Node(id="3", type=NodeType.LLM_ENGINE, data=NodeData(label="GPT-4", config={"system_prompt": "Be concise."}))
    node4 = Node(id="4", type=NodeType.OUTPUT, data=NodeData(label="Final Answer"))

    # 2. Create Mock Edges (Linear Flow)
    edges = [
        Edge(id="e1", source="1", target="2"),
        Edge(id="e2", source="2", target="3"),
        Edge(id="e3", source="3", target="4")
    ]

    # 3. Create Request
    request = WorkflowRequest(
        nodes=[node1, node2, node3, node4],
        edges=edges,
        user_query="How does this engine work?"
    )

    # 4. Run Engine
    print("Running Workflow Engine Test...")
    engine = WorkflowEngine(request)
    result = await engine.execute()
    
    # 5. Print Result
    print("\n--- Execution Result ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_engine())
