import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import gc
from typing import List, Dict, Any, Tuple
import logging
import os
from optimized_config import OptimizedConfig

class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db", 
                 batch_size: int = 32, enable_optimizations: bool = True):
        """Initialize RAG pipeline with embedding model and vector database"""
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.enable_optimizations = enable_optimizations
        
        # Initialize embedding cache for performance
        self.embedding_cache = {}
        self.max_cache_size = OptimizedConfig.MAX_CACHE_SIZE if enable_optimizations else 100
        
        # Initialize embedding model (lightweight, CPU-friendly)
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
        
        # Initialize ChromaDB with optimizations
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            if enable_optimizations:
                # Use optimized settings for better performance
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                self.client = chromadb.PersistentClient(path=persist_directory, settings=settings)
            else:
                self.client = chromadb.PersistentClient(path=persist_directory)
            
            self.collection = None
            self.logger.info("Initialized ChromaDB client with optimizations")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def create_collection(self, collection_name: str = "documents"):
        """Create or get a collection for storing document embeddings with optimizations"""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=collection_name)
            except:
                pass
            
            # Create collection with optimized settings
            metadata = {"hnsw:space": "cosine"}
            if self.enable_optimizations:
                # Only use supported HNSW parameters for ChromaDB
                metadata.update({
                    "hnsw:M": 16,  # Number of bi-directional links for each new element during construction
                })
            
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata
            )
            self.logger.info(f"Created optimized collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        if not self.collection:
            self.create_collection()
        
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            texts.append(doc['content'])
            metadatas.append({
                'document': doc['document'],
                'page_number': doc['page_number'],
                'section_title': doc.get('section_title', ''),
                'chunk_id': i
            })
            ids.append(f"doc_{i}")
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"Added {len(documents)} documents to collection")
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def search_similar(self, query: str, n_results: int = 10, 
                      persona_filter: str = None, job_filter: str = None) -> List[Dict]:
        """Search for similar documents based on query"""
        if not self.collection:
            raise ValueError("No collection initialized")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'document': results['metadatas'][0][i]['document'],
                    'page_number': results['metadatas'][0][i]['page_number'],
                    'section_title': results['metadatas'][0][i]['section_title']
                })
            
            self.logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_context_for_persona_job(self, persona: str, job: str, top_k: int = 5) -> List[Dict]:
        """Get relevant context based on persona and job requirements"""
        # Create a combined query from persona and job
        combined_query = f"{persona} {job}"
        
        # Search for relevant documents
        similar_docs = self.search_similar(combined_query, n_results=top_k * 2)
        
        # Score documents based on persona and job relevance
        scored_docs = []
        for doc in similar_docs:
            persona_score = self._calculate_persona_relevance(doc['content'], persona)
            job_score = self._calculate_job_relevance(doc['content'], job)
            combined_score = (persona_score + job_score) / 2
            
            doc['persona_relevance'] = persona_score
            doc['job_relevance'] = job_score
            doc['combined_score'] = combined_score
            scored_docs.append(doc)
        
        # Sort by combined score and return top_k
        scored_docs.sort(key=lambda x: x['combined_score'], reverse=True)
        return scored_docs[:top_k]
    
    def _calculate_persona_relevance(self, content: str, persona: str) -> float:
        """Calculate how relevant content is to the persona"""
        persona_keywords = self._extract_persona_keywords(persona)
        content_lower = content.lower()
        
        matches = sum(1 for keyword in persona_keywords if keyword in content_lower)
        return matches / len(persona_keywords) if persona_keywords else 0.0
    
    def _calculate_job_relevance(self, content: str, job: str) -> float:
        """Calculate how relevant content is to the job"""
        job_keywords = self._extract_job_keywords(job)
        content_lower = content.lower()
        
        matches = sum(1 for keyword in job_keywords if keyword in content_lower)
        return matches / len(job_keywords) if job_keywords else 0.0
    
    def _extract_persona_keywords(self, persona: str) -> List[str]:
        """Extract relevant keywords from persona description"""
        # This could be enhanced with NLP techniques
        common_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.lower().strip('.,!?') for word in persona.split()]
        return [word for word in words if word not in common_stopwords and len(word) > 2]
    
    def _extract_job_keywords(self, job: str) -> List[str]:
        """Extract relevant keywords from job description"""
        common_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.lower().strip('.,!?') for word in job.split()]
        return [word for word in words if word not in common_stopwords and len(word) > 2]
    
    def get_cached_embedding(self, text: str):
        """Get cached embedding or compute new one"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self.embedding_cache:
            self.embedding_cache[text_hash] = self.embedding_model.encode(text)
            
            # Manage cache size
            if len(self.embedding_cache) > self.max_cache_size:
                # Remove oldest entries
                keys_to_remove = list(self.embedding_cache.keys())[:len(self.embedding_cache)//2]
                for key in keys_to_remove:
                    del self.embedding_cache[key]
                    
        return self.embedding_cache[text_hash]
    
    def batch_process_documents(self, documents: List[str]) -> List[np.ndarray]:
        """Process documents in batches for better performance"""
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            # Check cache first for each document
            batch_embeddings = []
            uncached_docs = []
            uncached_indices = []
            
            for j, doc in enumerate(batch):
                text_hash = hashlib.md5(doc.encode()).hexdigest()
                if text_hash in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[text_hash])
                else:
                    uncached_docs.append(doc)
                    uncached_indices.append(j)
            
            # Process uncached documents
            if uncached_docs:
                new_embeddings = self.embedding_model.encode(uncached_docs)
                for k, embedding in enumerate(new_embeddings):
                    doc = uncached_docs[k]
                    text_hash = hashlib.md5(doc.encode()).hexdigest()
                    self.embedding_cache[text_hash] = embedding
                    
                    # Insert at correct position
                    original_index = uncached_indices[k]
                    batch_embeddings.insert(original_index, embedding)
            
            results.extend(batch_embeddings)
            
            # Force garbage collection after each batch
            if self.enable_optimizations:
                gc.collect()
        
        return results
    
    def optimize_memory(self):
        """Optimize memory usage by clearing caches"""
        if hasattr(self, 'embedding_cache'):
            cache_size = len(self.embedding_cache)
            if cache_size > self.max_cache_size:
                # Keep only the most recent half
                keys_to_remove = list(self.embedding_cache.keys())[:cache_size//2]
                for key in keys_to_remove:
                    del self.embedding_cache[key]
                self.logger.info(f"Cleared {len(keys_to_remove)} cached embeddings")
        
        gc.collect()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'embedding_cache_size': len(getattr(self, 'embedding_cache', {})),
            'batch_size': self.batch_size,
            'optimizations_enabled': self.enable_optimizations,
            'max_cache_size': self.max_cache_size
        }
