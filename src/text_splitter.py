from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
        separators=['\n\n', '\n', ' ', '']
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    return split_docs