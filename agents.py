import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load the environment variables (your API key) from the .env file
load_dotenv()

def test_connection():
    print("Initiating connection to Google AI Studio...")
    
    try:
        # 2. Initialize the model. We use 'gemini-1.5-flash' because it is 
        # incredibly fast and perfectly suited for the multi-agent loop later.
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=1.0 
        )

        # 3. Send a simple string to the model
        prompt = "Hello! Please reply with exactly: 'Connection successful! The model is active.'"
        print("Sending prompt...")
        response = llm.invoke(prompt)
        
        # 4. Print the result
        print("\n--- Response from Cloud LLM ---")
        print(response.content)
        print("-------------------------------")
        
    except Exception as e:
        print("\n❌ Connection Failed!")
        print(f"Error details: {e}")
        print("Double-check that your .env file is formatted correctly and the key is valid.")

if __name__ == "__main__":
    test_connection()