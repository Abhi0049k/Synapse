from src.doc_loader import process_all_docs
from src.text_splitter import split_documents
from src.embedding import EmbeddingManager
from src.rag_retriever import RAGRetriever
from src.vectordb import VectorStore
from src.rag_pipeline import RAGPipeline

def main():
    all_docs = process_all_docs("./files")
    chunks = split_documents(all_docs)
    texts = [doc.page_content for doc in chunks]
    embedding_manager = EmbeddingManager()
    vectors = embedding_manager.generate_embeddings(texts)
    vector_store = VectorStore()
    vector_store.add_documents(chunks, vectors)
    r_retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)
    r_pipeline = RAGPipeline(retriever=r_retriever, vector_store=vector_store, embedding_manager=embedding_manager)
    
    while True:
        user_query = input("Please enter what you want to search for or enter exit to end the simulation: ")
        if user_query=="exit" or user_query=="EXIT": break
        response = r_pipeline.query(user_query, top_k=3)
        print("\nFinal Answer:", response["answer"])
        print("Summary:", response["summary"])
        print("History:", response["history"][-1])
        print("\n\n")

if __name__ == "__main__":
    main()