from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from pathlib import Path

def process_all_docs(directory):
    documents = [];
    path = Path(directory)
    
    ### for PDF
    pdf_files = list(path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            document = loader.load()
            
            for doc in document:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
                
            documents.extend(document)
            print(f"Loaded {len(document)} pages")
        except Exception as e:
            print(f"Error: {e}")
    
    ### for CSV
    csv_files = list(path.glob("**/*.csv"))
    print(f"Found {len(csv_files)}")
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        try:
            loader = CSVLoader(str(csv_file))
            document = loader.load()
            # print(document)
            for doc in document:
                # print(doc,"\n")
                doc.metadata['source_file'] = csv_file.name
                doc.metadata['file_type'] = 'csv'

            documents.extend(document)
            print(f"Loaded {len(document)} pages")
        except Exception as e:
            print(f"Error: {e}")
    
    ### for Text
    text_files = list(path.glob("**/*.txt"))
    print(f"Found {len(text_files)}")
    for text_file in text_files:
        print(f"\nProcessing: {text_file.name}")
        try:
            loader = TextLoader(str(text_file))
            document = loader.load()
            for doc in document:
                doc.metadata['source_file'] = text_file.name
                doc.metadata['file_type'] = 'txt'
            documents.extend(document);
            print(f"Loaded {len(document)} page")
        except Exception as e:
            print(f"Error: {e}")
    
    return documents