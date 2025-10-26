import time
import os
from src.config.gemini_config import GeminiLLM
from langchain_core.documents import Document

# Gets the absolute path to this file's directory (.../RAG/src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Goes up one level to the root directory (.../RAG)
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# Builds the correct, absolute path to the log FILE
DEFAULT_LOG_FILE = os.path.join(ROOT_DIR, "files", "missed_queries.txt")

class RAGPipeline:
    def __init__(self, retriever, embedding_manager, vector_store, log_file=DEFAULT_LOG_FILE):
        try:
            self.retriever = retriever
            self.llm = GeminiLLM()
            self.history = []
            self.log_file = log_file
            self.embedding_manager = embedding_manager
            self.vector_store = vector_store
            
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            print("Pipeline setup successfully!")
        except Exception as e:
            print(f"Error while setting up the pipeline: {e}")
            raise
    
    def add_query_to_vector_store(self, query):
        """
        Logs a missed query to a common file and adds the query itself
        to the vector store as a new document
        """
        # Log to the common text file
        print("ADDING MISSED Queries")
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}]: {query}\n")
                
            metadata = {
                "source": "added_user_query",
                "timestamp":timestamp,
                "original_query": query
            }
            
            # Manually create the Document object
            new_doc = Document(page_content=query, metadata=metadata)
            
            # Manually generate the embedding for the query text
            vector = self.embedding_manager.generate_embeddings([query])[0]
            
            # Add to store using the same method as main.py
            self.vector_store.add_documents(documents=[new_doc], embeddings=[vector])
            print(f"Successfully add query to the vector store")
        except Exception as e:
            print(f"Failed to log missed query: {e}")
    
    def query(self, question, top_k=5, min_score=0.2, stream=False, summarize=False):
        results = self.retriever.retrieve(query = question, top_k=top_k, score_threshold=min_score)
        if not results:
            self.add_query_to_vector_store(question)
            answer = "Sorry, I couldn't find any relevant information to answer your question." 
            sources = []
            context = "" 
        else :
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [{
                "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
                "page": doc["metadata"].get("page", "unknown"),
                "score": doc["similarity_score"],
                "preview": doc["content"][:120] + "...."
            } for doc in results]
            
            # Streaming answer simulation
            
            prompt= f"""Use the following context to answer the question concisely. \nContext:\n{context}\n\nQuestion:{question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end="", flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.query([prompt.format(context=context, question=question)])
            answer = response.content
            
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n"+"\n".join(citations) if citations else answer
        
        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content 
            
        # Store query history
        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary
        })
        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history
        }