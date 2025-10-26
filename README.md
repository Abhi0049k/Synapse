# Synapse 🧠
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/) [![LangChain](https://img.shields.io/badge/langchain-blue?logo=langchain)](https://www.langchain.com/)

An intelligent, self-correcting Retrieval-Augmented Generation (RAG) pipeline built with Python, LangChain, and Google's Gemini API.

This project goes beyond a standard RAG implementation. It features an **active learning loop** that enables the system to learn from its mistakes and improve its knowledge base over time.

---

## Key Features

* **Retrieval-Augmented Generation (RAG):** Answers user questions by retrieving relevant context from a local document set.
* **Self-Correcting Knowledge Base:** When a user asks a question and no relevant documents are found, the system automatically:
    1.  **Logs** the "missed" query to a text file.
    2.  **Embeds** the user's query text.
    3.  **Adds** the query as a new document back into the vector store.
* **Constantly Evolving:** Because of this "active learning" feature, the application's knowledge base grows with every user interaction, allowing it to answer similar queries correctly in the future.

---

## Technology Stack

* **Language:** Python
* **Core Framework:** LangChain
* **LLM:** Google Gemini (via `langchain-google-genai`)
* **Vector Store:** ChromaDB
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)

---

## How It Works

1.  **Ingestion:** All documents in the `/files` directory are loaded, chunked, and embedded into a ChromaDB vector store.
2.  **Query:** A user asks a question.
3.  **Retrieve:** The system searches the vector store for relevant chunks of text.
4.  **Evaluate:**
    * **If Context is Found:** The context and the query are sent to the Gemini LLM to generate a factual answer.
    * **If No Context is Found:** The system logs the query to `files/missed_queries.txt`, then **embeds and adds the query itself** to the vector store. This allows it to be found in future, similar searches. The user receives a standard conversational response.
5.  **Repeat:** The application gets smarter with every missed query.

---

## 🚀 Setup and Installation

### 1. Clone the Repository

```sh
git clone https://github.com/Abhi0049k/Synapse.git
cd Synapse
```

### 2. Create a Virtual Environment
`uv venv` can create the virtual environment and install all packages from your `requirements.txt` and `uv.lock` files with just two commands.
```sh
# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```sh
uv pip sync requirements.txt
```

### 4 Set Up Your Environment
This is the most important step. Your API keys must be kept secret.
1. Create a file named `.env` in the root of the project.
2. Add your Google API key to it:
    ``` sh[Code snippet]
    GOOGLE_API_KEY=your_secret_api_key_here
    ```
3. This .env file is already listed in .gitignore, so it will never be uploaded to GitHub.

### 5. Add Your Documents
Place any PDF, TXT, or other documents you want the RAG to learn from into the `/files` directory.

--- 

## 💡 Usage
Run the main.py script to start the ingestion process and begin the chat session.
```sh
uv run main.py
```
The application will first process all your documents and build the vector store. Once complete, you will be prompted to ask questions:
```sh
Pipeline setup successfully!
Please enter what you want to search for or enter exit to end the simulation:
```
---
## 📜 License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.
