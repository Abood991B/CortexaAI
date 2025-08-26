"""Memory Manager with Vector Database and RAG Integration."""

from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
import logging


from config.config import (
    memory_config, get_logger, log_performance,
    cache_manager, generate_cache_key
)

# Set up structured logging
logger = get_logger(__name__)


class MemoryManager:
    """Advanced memory management with vector database and RAG capabilities."""

    def __init__(self):
        """Initialize the memory manager."""
        self.vector_store = None
        self.embedding_model = None
        self.knowledge_base = {}
        self.conversation_memories = {}
        self.memory_cache = {}

        # Initialize components
        self._initialize_vector_store()
        self._initialize_embedding_model()
        self._load_knowledge_base()

        # Metrics
        self._memory_operations = 0
        self._rag_queries = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def _initialize_vector_store(self):
        """Initialize the vector database based on configuration."""
        if not memory_config.enable_vector_db:
            logger.info("Vector database disabled in configuration")
            return

        try:
            if memory_config.vector_db_provider == 'chromadb':
                self.vector_store = ChromaVectorStore(
                    host=memory_config.vector_db_host,
                    port=memory_config.vector_db_port
                )
                logger.info("ChromaDB vector store initialized")

            elif memory_config.vector_db_provider == 'pinecone':
                self.vector_store = PineconeVectorStore(
                    api_key=memory_config.vector_db_api_key
                )
                logger.info("Pinecone vector store initialized")

            elif memory_config.vector_db_provider == 'weaviate':
                self.vector_store = WeaviateVectorStore(
                    host=memory_config.vector_db_host,
                    port=memory_config.vector_db_port
                )
                logger.info("Weaviate vector store initialized")

            else:
                logger.warning(f"Unknown vector DB provider: {memory_config.vector_db_provider}")
                self.vector_store = InMemoryVectorStore()

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = InMemoryVectorStore()
            logger.info("Fallback to in-memory vector store")

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            # For now, use a simple hash-based embedding for demonstration
            # In production, integrate with OpenAI, Cohere, or local models
            self.embedding_model = HashEmbeddingModel(
                dimensions=memory_config.embedding_dimensions
            )
            logger.info(f"Embedding model initialized with {memory_config.embedding_dimensions} dimensions")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def _load_knowledge_base(self):
        """Load knowledge base documents."""
        if not memory_config.enable_knowledge_base:
            logger.info("Knowledge base disabled in configuration")
            return

        try:
            # Load domain-specific knowledge
            self.knowledge_base = {
                'software_engineering': self._load_domain_knowledge('software_engineering'),
                'data_science': self._load_domain_knowledge('data_science'),
                'general': self._load_domain_knowledge('general')
            }
            logger.info("Knowledge base loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    def _load_domain_knowledge(self, domain: str) -> List[Dict[str, Any]]:
        """Load knowledge for a specific domain."""
        # This would load from files, databases, or APIs
        # For demonstration, return sample knowledge
        return [
            {
                'id': f'{domain}_1',
                'content': f'Sample knowledge for {domain}',
                'metadata': {'domain': domain, 'type': 'best_practice'},
                'timestamp': time.time()
            }
        ]

    async def store_memory(self, user_id: str, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store content in long-term memory with vector embedding.

        Args:
            user_id: User identifier
            content: Content to store
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        memory_id = self._generate_memory_id(user_id, content)

        # Generate embedding
        if self.embedding_model:
            embedding = await self.embedding_model.embed(content)
        else:
            embedding = None

        # Prepare memory entry
        memory_entry = {
            'id': memory_id,
            'user_id': user_id,
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'ttl': time.time() + (memory_config.memory_ttl_hours * 3600)
        }

        # Store in vector database
        if self.vector_store:
            await self.vector_store.store(memory_entry)

        # Cache for fast access
        self.memory_cache[memory_id] = memory_entry

        self._memory_operations += 1
        logger.info(f"Stored memory {memory_id} for user {user_id}")

        return memory_id

    async def retrieve_relevant_memories(self, user_id: str, query: str,
                                        top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using vector similarity search.

        Args:
            user_id: User identifier
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant memories
        """
        top_k = top_k or memory_config.rag_top_k

        # Check cache first
        cache_key = generate_cache_key(f"{user_id}:{query}:{top_k}", "memory_retrieve")
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            self._cache_hits += 1
            return cached_result

        self._cache_misses += 1

        # Generate query embedding
        if self.embedding_model and self.vector_store:
            query_embedding = await self.embedding_model.embed(query)

            # Search vector database
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=top_k,
                similarity_threshold=memory_config.rag_similarity_threshold
            )
        else:
            # Fallback to keyword search
            results = self._keyword_search(query, user_id, top_k)

        # Cache results
        cache_manager.set(cache_key, results, ttl=300)  # 5 minutes

        self._rag_queries += 1
        logger.info(f"Retrieved {len(results)} memories for user {user_id}")

        return results

    async def retrieve_knowledge(self, domain: str, query: str,
                                top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge from the knowledge base.

        Args:
            domain: Domain to search in
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant knowledge entries
        """
        top_k = top_k or memory_config.rag_top_k

        # Get domain knowledge
        domain_knowledge = self.knowledge_base.get(domain, [])

        # Simple text similarity search (in production, use embeddings)
        results = []
        query_lower = query.lower()

        for entry in domain_knowledge:
            content_lower = entry['content'].lower()
            if any(word in content_lower for word in query_lower.split()):
                results.append(entry)
                if len(results) >= top_k:
                    break

        logger.info(f"Retrieved {len(results)} knowledge entries for domain {domain}")
        return results

    async def generate_rag_context(self, user_id: str, domain: str,
                                  query: str) -> Dict[str, Any]:
        """
        Generate comprehensive context using RAG.

        Args:
            user_id: User identifier
            domain: Domain context
            query: Current query

        Returns:
            RAG context with memories and knowledge
        """
        # Retrieve relevant memories
        memories = await self.retrieve_relevant_memories(user_id, query)

        # Retrieve relevant knowledge
        knowledge = await self.retrieve_knowledge(domain, query)

        # Combine and rank results
        context_parts = []

        # Add memories with higher priority
        for memory in memories:
            context_parts.append({
                'type': 'memory',
                'content': memory['content'],
                'relevance': memory.get('score', 0.8),
                'timestamp': memory.get('timestamp')
            })

        # Add knowledge
        for entry in knowledge:
            context_parts.append({
                'type': 'knowledge',
                'content': entry['content'],
                'relevance': 0.7,
                'metadata': entry.get('metadata', {})
            })

        # Sort by relevance
        context_parts.sort(key=lambda x: x['relevance'], reverse=True)

        # Limit context length
        total_length = 0
        filtered_parts = []

        for part in context_parts:
            content_length = len(part['content'])
            if total_length + content_length <= memory_config.rag_max_context_length:
                filtered_parts.append(part)
                total_length += content_length
            else:
                break

        return {
            'query': query,
            'user_id': user_id,
            'domain': domain,
            'context_parts': filtered_parts,
            'total_context_length': total_length,
            'memories_count': len(memories),
            'knowledge_count': len(knowledge)
        }

    def update_conversation_memory(self, user_id: str, message: str,
                                 response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Update conversation memory for a user.

        Args:
            user_id: User identifier
            message: User message
            response: System response
            metadata: Additional metadata
        """
        if not memory_config.enable_conversation_memory:
            return

        if user_id not in self.conversation_memories:
            self.conversation_memories[user_id] = []

        # Add new turn
        turn = {
            'timestamp': time.time(),
            'message': message,
            'response': response,
            'metadata': metadata or {}
        }

        self.conversation_memories[user_id].append(turn)

        # Limit conversation history
        max_turns = memory_config.conversation_memory_max_turns
        if len(self.conversation_memories[user_id]) > max_turns:
            self.conversation_memories[user_id] = self.conversation_memories[user_id][-max_turns:]

        logger.debug(f"Updated conversation memory for user {user_id}")

    def get_conversation_context(self, user_id: str,
                               max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent conversation context for a user.

        Args:
            user_id: User identifier
            max_turns: Maximum number of turns to return

        Returns:
            List of conversation turns
        """
        if user_id not in self.conversation_memories:
            return []

        turns = self.conversation_memories[user_id]
        max_turns = max_turns or memory_config.conversation_memory_max_turns

        return turns[-max_turns:] if len(turns) > max_turns else turns

    def _generate_memory_id(self, user_id: str, content: str) -> str:
        """Generate a unique memory ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"mem_{user_id}_{content_hash}_{timestamp}"

    def _keyword_search(self, query: str, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback keyword-based search when embeddings are not available."""
        results = []
        query_lower = query.lower()

        # Search in cache
        for memory_id, memory in self.memory_cache.items():
            if memory['user_id'] == user_id:
                content_lower = memory['content'].lower()
                if any(word in content_lower for word in query_lower.split()):
                    results.append(memory)
                    if len(results) >= top_k:
                        break

        return results

    async def cleanup_expired_memories(self):
        """Clean up expired memories."""
        current_time = time.time()
        expired_keys = []

        # Find expired memories
        for memory_id, memory in self.memory_cache.items():
            if current_time > memory.get('ttl', 0):
                expired_keys.append(memory_id)

        # Remove expired memories
        for key in expired_keys:
            del self.memory_cache[key]

        # Clean vector store if available
        if self.vector_store:
            await self.vector_store.cleanup_expired()

        logger.info(f"Cleaned up {len(expired_keys)} expired memories")

    def get_metrics(self) -> Dict[str, Any]:
        """Get memory management metrics."""
        return {
            'memory_operations': self._memory_operations,
            'rag_queries': self._rag_queries,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            'cached_memories': len(self.memory_cache),
            'active_conversations': len(self.conversation_memories),
            'vector_store_available': self.vector_store is not None,
            'embedding_model_available': self.embedding_model is not None
        }


# Vector Store Implementations
class VectorStoreBase:
    """Base class for vector store implementations."""

    async def store(self, memory_entry: Dict[str, Any]):
        """Store a memory entry."""
        raise NotImplementedError

    async def search(self, query_embedding: List[float], user_id: str,
                    top_k: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        raise NotImplementedError

    async def cleanup_expired(self):
        """Clean up expired entries."""
        raise NotImplementedError


class InMemoryVectorStore(VectorStoreBase):
    """Simple in-memory vector store for development and testing."""

    def __init__(self):
        self.memories = {}

    async def store(self, memory_entry: Dict[str, Any]):
        """Store memory in memory."""
        self.memories[memory_entry['id']] = memory_entry

    async def search(self, query_embedding: List[float], user_id: str,
                    top_k: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Simple keyword-based search."""
        results = []
        for memory in self.memories.values():
            if memory['user_id'] == user_id:
                # Simple similarity score based on content overlap
                score = 0.5  # Placeholder
                if score >= similarity_threshold:
                    memory_copy = memory.copy()
                    memory_copy['score'] = score
                    results.append(memory_copy)
                    if len(results) >= top_k:
                        break

        return results

    async def cleanup_expired(self):
        """Clean up expired memories."""
        current_time = time.time()
        expired_ids = [
            mem_id for mem_id, memory in self.memories.items()
            if current_time > memory.get('ttl', 0)
        ]
        for mem_id in expired_ids:
            del self.memories[mem_id]


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB vector store implementation."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.client = None
        self.collection = None

    async def _initialize(self):
        """Initialize ChromaDB connection."""
        try:
            import chromadb
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
            self.collection = self.client.get_or_create_collection("memories")
        except ImportError:
            logger.error("ChromaDB not installed. Install with: pip install chromadb")
            raise

    async def store(self, memory_entry: Dict[str, Any]):
        """Store memory in ChromaDB."""
        if not self.client:
            await self._initialize()

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.collection.add(
                ids=[memory_entry['id']],
                embeddings=[memory_entry['embedding']],
                metadatas=[memory_entry['metadata']],
                documents=[memory_entry['content']]
            )
        )

    async def search(self, query_embedding: List[float], user_id: str,
                    top_k: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar memories."""
        if not self.client:
            await self._initialize()

        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"user_id": user_id}
            )
        )

        # Process results
        processed_results = []
        for i, doc in enumerate(results['documents'][0]):
            score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
            if score >= similarity_threshold:
                processed_results.append({
                    'id': results['ids'][0][i],
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'score': score
                })

        return processed_results

    async def cleanup_expired(self):
        """Clean up expired entries."""
        # Implementation would depend on ChromaDB's capabilities
        pass


class PineconeVectorStore(VectorStoreBase):
    """Pinecone vector store implementation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.index = None

    async def _initialize(self):
        """Initialize Pinecone connection."""
        try:
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index("memories")
        except ImportError:
            logger.error("Pinecone not installed. Install with: pip install pinecone-client")
            raise

    async def store(self, memory_entry: Dict[str, Any]):
        """Store memory in Pinecone."""
        if not self.index:
            await self._initialize()

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.index.upsert([
                {
                    'id': memory_entry['id'],
                    'values': memory_entry['embedding'],
                    'metadata': {
                        **memory_entry['metadata'],
                        'user_id': memory_entry['user_id'],
                        'content': memory_entry['content'],
                        'timestamp': memory_entry['timestamp']
                    }
                }
            ])
        )

    async def search(self, query_embedding: List[float], user_id: str,
                    top_k: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Search Pinecone for similar memories."""
        if not self.index:
            await self._initialize()

        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter={"user_id": user_id},
                include_metadata=True
            )
        )

        # Process results
        processed_results = []
        for match in results['matches']:
            score = match['score']
            if score >= similarity_threshold:
                processed_results.append({
                    'id': match['id'],
                    'content': match['metadata']['content'],
                    'metadata': match['metadata'],
                    'score': score
                })

        return processed_results

    async def cleanup_expired(self):
        """Clean up expired entries."""
        # Implementation would depend on Pinecone's capabilities
        pass


class WeaviateVectorStore(VectorStoreBase):
    """Weaviate vector store implementation."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.client = None

    async def _initialize(self):
        """Initialize Weaviate connection."""
        try:
            import weaviate
            self.client = weaviate.Client(f"http://{self.host}:{self.port}")
        except ImportError:
            logger.error("Weaviate client not installed. Install with: pip install weaviate-client")
            raise

    async def store(self, memory_entry: Dict[str, Any]):
        """Store memory in Weaviate."""
        if not self.client:
            await self._initialize()

        # Implementation would depend on Weaviate schema
        pass

    async def search(self, query_embedding: List[float], user_id: str,
                    top_k: int, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Search Weaviate for similar memories."""
        if not self.client:
            await self._initialize()

        # Implementation would depend on Weaviate schema
        return []

    async def cleanup_expired(self):
        """Clean up expired entries."""
        pass


class HashEmbeddingModel:
    """Simple hash-based embedding model for demonstration."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    async def embed(self, text: str) -> List[float]:
        """Generate embedding using hash functions."""
        # Simple hash-based embedding for demonstration
        import hashlib

        # Generate multiple hashes for different dimensions
        embedding = []
        for i in range(self.dimensions // 32):
            hash_obj = hashlib.md5(f"{text}:{i}".encode())
            hash_int = int(hash_obj.hexdigest(), 16)

            # Convert to 32 float values between -1 and 1
            for j in range(32):
                bit_value = (hash_int >> j) & 1
                float_value = 1.0 if bit_value else -1.0
                embedding.append(float_value)

        return embedding[:self.dimensions]


# Global memory manager instance
memory_manager = MemoryManager()
