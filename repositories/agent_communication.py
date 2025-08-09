from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rich import print
from vectordboperation.vectorEntity import retriever


def sendTestMessageToModle(modelName:str="llama3.2", message:str="Hello, AI!"):
    """
    This function sends a message to the model.
    """
    # Initialize the OllamaLLM with the specified model
    model= OllamaLLM(model=modelName)

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
    return result


def sendMessageToModle(modelName:str="llama3.2", message:str="Hello, AI!"):
    """
    This function sends a message to the model.
    """
    # Initialize the OllamaLLM with the specified model
    model= OllamaLLM(model=modelName)

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
    print("[bold yellow]Processing your query...[/bold yellow]")
    print(message)
    relevant_docs = retriever.invoke(message)
    information = "\n".join([doc.page_content for doc in relevant_docs])
    response = chain.invoke({"information": information, "query": message})
    print(f"[bold green]Response: {response}[/bold green]")


    return {
            "query": message,
            "model": modelName,
            "response": response
        }


    # Create a PromptTemplate instance with the defined template
    prompt = PromptTemplate.from_template(template)
    # Combine the prompt and model into a chain
    chain = prompt | model

    # Example usage of the chain to invoke a response
    result = chain.invoke({"information": "Llama 3.1 is a state-of-the-art language model.", "query": "What is Llama 3.1?"})

    # Print the result of the chain invocation
    print(result)
    print ("[bold green]AI Agent initialized successfully![/bold green]")
    return result