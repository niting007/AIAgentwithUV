from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rich import print
from vectordboperation.vectorEntity import retriever
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.chatagent import router as chat_router



app = FastAPI()
# Initialize the FastAPI application with CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include the chat agent router for handling chat-related endpoints
app.include_router(chat_router, prefix="/agent", tags=["Agent Chat"])

# Define the root endpoint for the FastAPI application
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Agent with Vector Database!"}

# Initialize the OllamaLLM with the specified model
model= OllamaLLM(model="llama3.2")

# Define the prompt template for the AI agent
template="""

You are a helpful AI agent. Your task is to assist users with their queries.

here are some relvelant information: {information}

Here is the user's query: {query}

Your response should be concise and directly address the user's query based on the provided information.

Respond in a friendly and professional manner.
"""

# Create a PromptTemplate instance with the defined template
prompt = PromptTemplate.from_template(template)
# Combine the prompt and model into a chain
chain = prompt | model

# Example usage of the chain to invoke a response
result = chain.invoke({"information": "Llama 3.1 is a state-of-the-art language model.", "query": "What is Llama 3.1?"})

# Print the result of the chain invocation
print(result)
print ("[bold green]AI Agent initialized successfully![/bold green]")
# # List all models currently installed locally
# models = ollama.list()
# print(models)
# This script initializes an AI agent that uses a vector database to answer user queries.
# Uncomment the following lines to run the AI agent in a loop for user interaction
# while True:
#     print("\n\n[bold green]Welcome to the AI Agent with Vector Database![/bold green]")
#     user_query = input("[bold blue]Enter your query (or 'q' to quit): [/bold blue]")
#     if user_query.lower() == 'q':
#         print("[bold red]Exiting the AI Agent. Goodbye![/bold red]")
#         break
#     try:
#         print("[bold yellow]Processing your query...[/bold yellow]")
#         print(user_query)
#         relevant_docs = retriver.invoke(user_query)
#         information = "\n".join([doc.page_content for doc in relevant_docs])
#         response = chain.invoke({"information": information, "query": user_query})
#         print(f"[bold green]Response: {response}[/bold green]")
    
#     except Exception as e:
#         print(f"[bold red]An error occurred: {e}[/bold red]")

# This is a simple main function to demonstrate the structure of the code.


def main():
    print("Hello from aiagentwithuv!")
