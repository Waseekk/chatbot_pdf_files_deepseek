# vectordb/__init__.py
# Empty file to mark directory as a package

# vectordb/faiss_store.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

class PDFVectorStore:
    """Handles vector storage and retrieval for PDF documents using FAISS."""
    
    def __init__(self, embedding_dim=384, index_path="./faiss_index"):
        """Initialize the vector store."""
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "index.faiss")
        self.metadata_file = os.path.join(index_path, "metadata.pkl")
        
        # Create embedding model using HuggingFace
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Load existing index or create new one
        if os.path.exists(self.index_file):
            self.load_index()
        else:
            self.create_index()
    
    def create_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.document_chunks = []
    
    def load_index(self):
        """Load existing FAISS index from disk."""
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.document_chunks = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading FAISS index: {str(e)}")
            self.create_index()
    
    def save_index(self):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.document_chunks, f)
        except Exception as e:
            st.error(f"Error saving FAISS index: {str(e)}")
    
    def process_document(self, document_name: str, text: str):
        """Process document text into chunks and add to vector store."""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Process each chunk to extract page numbers
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            # Extract page information from chunk using regex pattern
            page_matches = re.findall(r'\[Page (\d+)\]', chunk)
            page_num = int(page_matches[-1]) if page_matches else 0
            
            metadata = {
                "document": document_name,
                "page": page_num,
                "chunk_id": i,
                "text": chunk
            }
            chunk_metadata.append(metadata)
        
        # Generate embeddings for chunks
        embeddings = self.embedding_model.embed_documents(chunks)
        
        # Add to FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        
        # Add chunk metadata
        self.document_chunks.extend(chunk_metadata)
        
        # Save updated index
        self.save_index()
        
        return len(chunks)
    
    def search(self, query: str, k: int = 5):
        """Search for similar chunks to the query."""
        if self.index.ntotal == 0:
            return []
        
        # Generate embedding for query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Convert to numpy array
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_np, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and 0 <= idx < len(self.document_chunks):
                result = self.document_chunks[idx].copy()
                result["score"] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def delete_document(self, document_name: str):
        """Remove a document from the vector store."""
        # This is a simplified implementation that recreates the index
        # A more sophisticated approach would use IDs to selectively delete
        
        # Filter out chunks from the specified document
        new_chunks = [chunk for chunk in self.document_chunks if chunk["document"] != document_name]
        
        # If no change, return early
        if len(new_chunks) == len(self.document_chunks):
            return 0
        
        # Create a new index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # If there are remaining documents, add them back
        if new_chunks:
            # Extract text from each chunk
            texts = [chunk["text"] for chunk in new_chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Add to FAISS index
            embeddings_np = np.array(embeddings).astype('float32')
            self.index.add(embeddings_np)
        
        # Update chunks and save
        self.document_chunks = new_chunks
        self.save_index()
        
        return len(self.document_chunks) - len(new_chunks)