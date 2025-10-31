import os
import json
import faiss
import numpy as np
import requests  # Using requests directly for stability
import time
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv  # To securely load the API key

# --- 1. CONFIGURATION ---
load_dotenv()  # Load variables from .env file

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # This error will show in your terminal if the key is missing
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file and add your key.")

KNOWLEDGE_BASE_DIR = "knowledge_base"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_STORE_FILE = "vector_store.index"
DOCUMENTS_FILE = "documents.json"

# This is the stable, direct API endpoint that works.
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- 2. RAG PIPELINE CLASS ---

class ServiceDeskRAG:
    """
    This class implements the RAG pipeline using direct API calls.
    It securely loads the API key, builds a vector store,
    and uses the stable Gemini API for generation.
    """

    def __init__(self):
        print("Initializing RAG pipeline...")
        self.documents = []
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        if os.path.exists(VECTOR_STORE_FILE) and os.path.exists(DOCUMENTS_FILE):
            print(f"Loading existing vector store from {VECTOR_STORE_FILE}")
            self.index = faiss.read_index(VECTOR_STORE_FILE)
            with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        else:
            print("No vector store found. Building a new one...")
            self._build_vector_store()

    def _load_documents(self):
        """Loads text files from the knowledge base directory."""
        documents_list = []
        # Sort the files to ensure consistent ordering for indexing
        filenames = sorted([f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".txt")])
        for filename in filenames:
            filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                documents_list.append(f.read())
        return documents_list

    def _build_vector_store(self):
        """Builds the FAISS vector index from the loaded documents."""
        self.documents = self._load_documents()
        
        if not self.documents:
            print("No documents found in knowledge_base/. Chatbot will not be able to answer questions.")
            return

        print(f"Found {len(self.documents)} documents. Embedding...")
        
        embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        print(f"Saving vector store to {VECTOR_STORE_FILE}...")
        faiss.write_index(self.index, VECTOR_STORE_FILE)
        
        print(f"Saving document texts to {DOCUMENTS_FILE}...")
        with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)
            
        print("Vector store built successfully.")

    def _retrieve_context(self, query_text, k=2):
        """Retrieves the top-k most relevant document chunks for a query."""
        if self.index is None:
            return [], []
        
        query_embedding = self.embedding_model.encode([query_text]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_chunks = [self.documents[i] for i in indices[0]]
        
        # Get the source filenames
        source_filenames = sorted([f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".txt")])
        source_files = [source_filenames[i] for i in indices[0] if i < len(source_filenames)]
            
        return retrieved_chunks, source_files

    def _call_gemini_with_backoff(self, payload, retries=5, delay=2):
        """Calls the Gemini API using requests with exponential backoff."""
        headers = {"Content-Type": "application/json"}
        
        for i in range(retries):
            try:
                response = requests.post(GEMINI_API_URL, data=json.dumps(payload), headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # This will print the *real* error if it's not a 404
                    print(f"API Error: Status {response.status_code}, Response: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request Error: {e}")
            
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
            
        return None # Failed after all retries

    def query(self, user_question):
        """
        The main RAG query function.
        1. Retrieves context. 2. Augments prompt. 3. Generates answer.
        """
        print(f"\nReceived query: {user_question}")
        
        # 1. Retrieve
        # --- THIS IS THE FIX ---
        # Changed k=2 to k=1 to be more precise and avoid irrelevant sources.
        context_chunks, source_files = self._retrieve_context(user_question, k=1)
        # --- END OF FIX ---
        
        if not context_chunks:
            return "I'm sorry, I could not find any relevant information.", []

        context_str = "\n\n---\n\n".join(context_chunks)
        
        # 2. Augment (Create the prompt and payload)
        system_prompt = (
            "You are a helpful and professional IT Service Desk assistant."
            "You must answer the user's question *only* using the context provided."
            "Do not make up information or answer questions not found in the context."
            "If the context does not contain the answer, you MUST say:"
            "'I'm sorry, my knowledge base does not have the information to answer this question. "
            "Please contact a human IT support agent.'"
        )
        
        user_prompt = f"""
        CONTEXT:
        ---
        {context_str}
        ---

        USER QUESTION:
        {user_question}
        """
        
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"parts": [{"text": user_prompt}]}
            ]
        }
        
        # 3. Generate
        print("Calling Gemini API...")
        try:
            response = self._call_gemini_with_backoff(payload)
            
            if response and response.get("candidates"):
                answer = response["candidates"][0]["content"]["parts"][0]["text"]
                return answer, source_files
            else:
                print(f"Unexpected API response: {response}")
                return "I'm sorry, I received an invalid response from the AI model.", []

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "I'm sorry, I encountered an error while generating a response.", []


