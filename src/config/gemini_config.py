import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

class GeminiLLM:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash-lite"):
        """
        Initialize Google Gemini LLM

        Args:
            api_key (str, optional): Google Gemini API Key Defaults to "None"
            model_name (str, optional): Good model  Defaults to "gemini-2.5-flash-lite".
        """
        self.model = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google Gemini API Key is required. Set GOOGLE_API_KEY environment variable")
        
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            temperature=0.1,
            model=self.model,
            model_kwargs={"max_token": 1024}
        )
        
        print(f"Initialized Google Gemini LLM with model: {self.model}")
    
    def query(self, query):
        try:
            response = self.llm.invoke(query)
            return response
        except Exception as e:
            print(f"Error during LLM Invocation: {e}")