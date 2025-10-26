from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """Handles document embedding generation using Sentence Transformer"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the Sentence Transformer model"""
        try:
            print(f"Loading embedding model: ", {self.model_name})
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error: {e}")
            raise # re-raising the error
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list texts

        Args:
            texts: List of text string to embed
        
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings