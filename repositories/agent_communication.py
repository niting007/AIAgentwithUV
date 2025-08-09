from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rich import print

from rich import print
# from vectordboperation.vectorEntity import retriever
from repositories.vectordb_manager import ChromaDBManager
from langchain.chains import LLMChain


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

def sendMessageToModel(modelName: str = "llama3.2", message: str = "Hello, AI!"):
    """
    Send a message to the model with context from Chroma retriever.
    """
    # Initialize the OllamaLLM with the specified model
    model = OllamaLLM(model=modelName)

    # Define the prompt template
    template = """
    You are a helpful AI agent. Your task is to assist users with their queries.

    Here are some relevant information: {information}

    Here is the user's query: {query}

    Your response should be concise and directly address the user's query based on the provided information.

    Respond in a friendly and professional manner.
    """
    try:
        
        prompt = PromptTemplate(template=template, input_variables=["information", "query"])

        # Create a chain from prompt and model
        chain = prompt | model

        # Initialize your ChromaDBManager (or reuse a singleton instance)
        manager = ChromaDBManager()

        # Get retriever and query relevant docs
        retriever = manager.get_retriever()
        if retriever is None:
            raise ValueError("Retriever not initialized or vector store not loaded.")

        relevant_docs = retriever.get_relevant_documents(message)

        # Join retrieved docs contents as context
        information = "\n".join(doc.page_content for doc in relevant_docs)

        print("[bold yellow]Processing your query...[/bold yellow]")
        print(message)

        # Run the chain with prompt variables
        response = chain.run({"information": information, "query": message})

        print(f"[bold green]Response: {response}[/bold green]")

        return {
            "query": message,
            "model": modelName,
            "response": response
        }
    except Exception as e:
        print(f"[bold red]An error occurred: {e}[/bold red]")
        return {
            "error": str(e),
            "query": message,
            "model": modelName
        }


# def get_response_from_vector_db_llama(modelName: str = "llama3.2",message: str= "Hello, AI!"):
#     try:

#         # Initialize your vector DB manager (ideally reuse singleton in real app)
#         manager = ChromaDBManager()

#         # Get the retriever, or error if not ready
#         retriever = manager.get_retriever()

#         # Check if retriever is initialized
#         if retriever is None:
#             raise ValueError("Retriever not initialized or vector store not loaded.")
        
#         # Retrieve relevant documents from vector DB
#         relevant_docs = retriever.get_relevant_documents(message)

#         # Combine document texts as context
#         context = "\n".join(doc.page_content for doc in relevant_docs)

#         # Define prompt template with placeholders
#         prompt_template = """
#         You are a helpful AI assistant.

#         Here is some relevant information extracted from a vector database:

#         {description}

#         User query: {query}

#         Please answer the query concisely and accurately using the above information.
#         """

#         prompt = PromptTemplate(
#             input_variables=["description", "query"],
#             template=prompt_template
#         )

#         # Initialize LLM with the specified model
#         print("[bold yellow]Initializing LLM with model:[/bold yellow]", modelName)
#         llm = OllamaLLM(model=modelName)
        

#         # Build chain with prompt and LLM
#         chain = LLMChain(llm=llm, prompt=prompt)

#         # Run chain with inputs
#         response = chain.run({"description": context, "query": message})

#         return response
#     except Exception as e:
#         print(f"[bold red]Error in get_response_from_vector_db_llama: {e}[/bold red]")
#         return {
#             "error": str(e),
#             "query": message,
#             "model": modelName
#         }



# def sendMessageToModle(modelName:str="llama3.2", message:str="Hello, AI!"):
#     """
#     This function sends a message to the model.
#     """
#     # Initialize the OllamaLLM with the specified model
#     model= OllamaLLM(model=modelName)

#     # Define the prompt template for the AI agent
#     template="""

#     You are a helpful AI agent. Your task is to assist users with their queries.

#     here are some relvelant information: {information}

#     Here is the user's query: {query}

#     Your response should be concise and directly address the user's query based on the provided information.

#     Respond in a friendly and professional manner.
#     """

#      # Create a PromptTemplate instance with the defined template
#     prompt = PromptTemplate.from_template(template)

#     # Combine the prompt and model into a chain
#     chain = prompt | model

#     # Example usage of the chain to invoke a response
#     print("[bold yellow]Processing your query...[/bold yellow]")
#     print(message)
#     relevant_docs = ChromaDBManager.get_retriever.invoke(message)
#     information = "\n".join([doc.page_content for doc in relevant_docs])
#     response = chain.invoke({"information": information, "query": message})
#     print(f"[bold green]Response: {response}[/bold green]")


#     return {
#             "query": message,
#             "model": modelName,
#             "response": response
#         }


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