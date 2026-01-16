from typing import List, Dict, Generator
import networkx as nx
import asyncio
import os
from groq import Groq
from schema import WorkflowRequest, NodeType
from services import query_knowledge_base

class WorkflowEngine:
    def __init__(self, workflow: WorkflowRequest):
        self.nodes = {node.id: node for node in workflow.nodes}
        self.edges = workflow.edges
        self.query = workflow.user_query
        self.context = {} 
        self.graph = self._build_graph()
        
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
        else:
            self.groq_client = None

    def _build_graph(self):
        G = nx.DiGraph()
        for node in self.nodes.values():
            G.add_node(node.id, type=node.type)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target)
        return G

    def validate_graph(self):
        """Checks for cycles and connectivity"""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph contains cycles")
        return True

    def get_execution_plan(self) -> List[str]:
        """Returns ordered list of Node IDs (Topological Sort)"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph contains cycles, cannot sort.")

    def stream_execution(self) -> Generator[str, None, None]:
        """
        Executes the workflow graph linearly and yields chunks for SSE.
        Synchronous generator for FastAPI StreamingResponse.
        """
        try:
            self.validate_graph()
            execution_plan = self.get_execution_plan()
        except ValueError as e:
            yield f"Error: {str(e)}"
            return

        results = {}
        
        for node_id in execution_plan:
            node = self.nodes[node_id]
            node_type = node.type
            
            # Yield status update
            yield f"status: Executing {node.data.label} ({node_type})\n\n"

            if node_type == NodeType.INPUT:
                results['query'] = self.query

            elif node_type == NodeType.KNOWLEDGE_BASE:
                collection_id = node.data.config.get('vector_collection_id')
                print(f"DEBUG: Knowledge Base Node - collection_id: {collection_id}")
                print(f"DEBUG: Node config: {node.data.config}")
                
                if collection_id:
                    yield "status: Querying knowledge base...\n\n"
                    # Query ChromaDB with user query
                    user_query = results.get('query', '')
                    print(f"DEBUG: Querying with: {user_query}")
                    context = query_knowledge_base(collection_id, user_query, top_k=3)
                    print(f"DEBUG: Retrieved context: {context[:200]}...")
                    results['context'] = context
                else:
                    yield "status: No document uploaded to knowledge base\n\n"
                    results['context'] = "No document has been uploaded to this knowledge base node."

            elif node_type == NodeType.LLM_ENGINE:
                if not self.groq_client:
                    yield "status: Error - GROQ_API_KEY not found\n\n"
                    results['llm_response'] = "Configuration Error: GROQ_API_KEY is missing."
                    continue

                prompt = node.data.config.get('system_prompt', 'You are a helpful assistant.')
                context = results.get('context', '')
                user_msg = results.get('query', '')
                
                # Construct messages
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Context: {context}\n\nUser Query: {user_msg}"}
                ]
                
                model = node.data.config.get('model', 'llama-3.3-70b-versatile')
                # Map generic names to Groq models if needed
                if "gpt" in model.lower(): 
                    model = "llama-3.3-70b-versatile" 

                yield f"status: Generating response with {model}...\n\n"
                
                try:
                    completion = self.groq_client.chat.completions.create(
                        messages=messages,
                        model=model,
                        stream=True
                    )

                    response_text = ""
                    for chunk in completion:
                        content = chunk.choices[0].delta.content
                        if content:
                            response_text += content
                            # Format for SSE (data: <content>)
                            # We replace newlines to keep the stream clean or handle on frontend
                            clean_content = content.replace('\n', '\\n') 
                            yield f"data: {clean_content} \n\n"
                    
                    results['llm_response'] = response_text
                
                except Exception as e:
                    yield f"status: LLM Error - {str(e)}\n\n"
                    results['llm_response'] = f"Error generating response: {str(e)}"

            elif node_type == NodeType.OUTPUT:
                final = results.get('llm_response', "No response")
                yield f"final: {final}\n\n"