import hashlib
import os
import uuid
from fastapi import UploadFile
from sqlmodel import Session
from schema import Document
import pdfplumber

# Mock embedding model for Vercel compatibility
class MockEmbeddingModel:
    def encode(self, text):
        return [0.0] * 384  # Return dummy embedding

embedding_model = MockEmbeddingModel()

# Mock ChromaDB client for Vercel compatibility
class MockChromaClient:
    def __init__(self):
        self.collections = {}
    
    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection()
        return self.collections[name]
    
    def get_collection(self, name):
        """Get existing collection or return empty mock collection"""
        return self.collections.get(name, MockCollection())

class MockCollection:
    def __init__(self):
        self.data = {}
    
    def add(self, ids, documents, embeddings, metadatas=None):
        """Store chunks with their embeddings"""
        for i, doc_id in enumerate(ids):
            self.data[doc_id] = {
                "document": documents[i] if i < len(documents) else "",
                "embedding": embeddings[i] if i < len(embeddings) else None,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {}
            }
    
    def query(self, query_embeddings, n_results):
        """Return dummy results matching ChromaDB format"""
        if not self.data:
            return {"documents": [[]], "metadatas": [[]]}
        
        # Return all stored documents (simplified - no actual embedding similarity)
        docs = [item["document"] for item in self.data.values()][:n_results]
        metadata = [item["metadata"] for item in self.data.values()][:n_results]
        
        return {"documents": [docs], "metadatas": [metadata]}

chroma_client = MockChromaClient()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks

class IngestionService:
    def __init__(self, db: Session):
        self.db = db

    async def ingest_document(self, file: UploadFile) -> Document:
        """
        Process uploaded document:
        1. Save file
        2. Extract text
        3. Chunk text
        4. Generate embeddings
        5. Store in ChromaDB
        6. Save metadata to DB
        """
        # Read file content
        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()

        # Check for duplicates
        existing = self.db.query(Document).filter(Document.file_hash == file_hash).first()
        if existing:
            return existing

        # Save file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(content)

        # Extract text from PDF
        try:
            full_text = extract_text_from_pdf(file_path)
            
            # Validate extracted text
            if not full_text or not full_text.strip():
                raise Exception("No text could be extracted from the PDF. The file might be image-based or empty.")
            
            chunks = chunk_text(full_text)
            
            # Validate chunks
            if not chunks or len(chunks) == 0:
                raise Exception("Failed to create text chunks from the PDF.")
            
            print(f"Extracted {len(full_text)} characters, created {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = embedding_model.encode(chunks).tolist()
            
            # Validate embeddings
            if not embeddings or len(embeddings) == 0:
                raise Exception("Failed to generate embeddings from text chunks.")
            
            # Create or get ChromaDB collection (use hash-based naming for consistency)
            collection_name = f"col_{file_hash}"
            collection = chroma_client.get_or_create_collection(name=collection_name)
            
            # Store chunks and embeddings in ChromaDB
            collection.add(
                ids=[f"chunk_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"chunk_index": i, "source": filename} for i in range(len(chunks))]
            )
            
            print(f"Successfully created collection: {collection_name} with {len(chunks)} chunks")
            
            # Save document metadata
            doc = Document(
                filename=file.filename,
                storage_path=file_path,
                file_hash=file_hash,
                vector_collection_id=collection_name
            )
            self.db.add(doc)
            self.db.commit()
            self.db.refresh(doc)
            
            return doc
            
        except Exception as e:
            # Clean up file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(f"Failed to process PDF: {str(e)}")

def query_knowledge_base(collection_name: str, query: str, top_k: int = 3) -> str:
    """
    Query ChromaDB collection with a question and return relevant context
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Combine retrieved chunks
        if results and results['documents']:
            context = "\n\n".join(results['documents'][0])
            return context
        else:
            return "No relevant information found in the document."
            
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"
