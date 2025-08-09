from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rich import print
from vectorEntity import retriver
model= OllamaLLM(model="llama3.2")

template="""

You are a helpful AI agent. Your task is to assist users with their queries.

here are some relvelant information: {information}

Here is the user's query: {query}

Your response should be concise and directly address the user's query based on the provided information.

Respond in a friendly and professional manner.
"""

prompt = PromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke({"information": "Llama 3.1 is a state-of-the-art language model.", "query": "What is Llama 3.1?"})

print(result)

while True:
    print("\n\n[bold green]Welcome to the AI Agent with Vector Database![/bold green]")
    user_query = input("[bold blue]Enter your query (or 'q' to quit): [/bold blue]")
    if user_query.lower() == 'q':
        print("[bold red]Exiting the AI Agent. Goodbye![/bold red]")
        break
    try:
        print("[bold yellow]Processing your query...[/bold yellow]")
        print(user_query)
        relevant_docs = retriver.invoke(user_query)
        information = "\n".join([doc.page_content for doc in relevant_docs])
        response = chain.invoke({"information": information, "query": user_query})
        print(f"[bold green]Response: {response}[/bold green]")
    
    except Exception as e:
        print(f"[bold red]An error occurred: {e}[/bold red]")

# This is a simple main function to demonstrate the structure of the code.


def main():
    print("Hello from aiagentwithuv!")