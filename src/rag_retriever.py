class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store, embedding_manager):
        """
        Initialize the retriever

        Args:
            vector_store (_type_): Vector store containing document embeddings
            embedding_manager (_type_): Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
    
    def retrieve(self, query, top_k = 5, score_threshold=0.0):
        """
        Retrieve relevant documents for a query

        Args:
            query (_type_): The search query
            top_k (int, optional): Number of top results to return. Defaults to 5.
            score_threshold (float, optional): Minimum similarity score threshold. Defaults to 0.0.
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            # Process results
            
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents =results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (chromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i+1
                        })
                        print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
                    else:
                        print("No documents found")
                    return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []