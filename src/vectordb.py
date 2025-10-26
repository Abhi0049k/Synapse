import os
import uuid
import chromadb
# from chromadb.config import Settings

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name="docs", persist_directory="data/vector_store"):
        """
        Initialize the vector store

        Args:
            collection_name (str, optional): _description_. Defaults to "docs", Name of the ChromaDB collection.
            persist_directory (str, optional): _description_. Defaults to "../data/vector_store", Directory to persist the vector store.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.initialize_store()
        
    def initialize_store(self):
        """Initialize ChromaDB Client and collection"""
        try:
            ### Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            ### Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document Embedding for RAG"}
            )
            
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents, embeddings):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents (List[Any]): List of Langchain documents
            embeddings (np.ndarray): Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        ### Prepare data for chromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique Id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_text.append(doc.page_content)
            
            embeddings_list.append(embedding.tolist())
            
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise