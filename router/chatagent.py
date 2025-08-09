from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import repositories.agent_metadata
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate

router = APIRouter()

# Import necessary modules for the chat agent
from vectordboperation.vectorEntity import retriever

# Define request body model
class QueryRequest(BaseModel):
    query: str
    model: str  # Added model parameter

# Define the router for the chat agent

@router.get("/models")
async def get_models():
    """
    Fetches the list of models currently installed locally using the Ollama API.
    
    Returns:
        list: A list of model names currently available.
    """
    try:
        models = repositories.agent_metadata.getModels()
        # Remove version/tag part from the name
        model_names = [m.split(":")[0] for m in models]
        return {"models": model_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_agent(request: QueryRequest):
    """
    Processes a user query using the specified model and returns the response.
    Args:
        request (QueryRequest): The request body containing the user's query and model name.
    Returns:
        dict: A dictionary containing the user's query, selected model, and the response from the AI agent.
    Raises:
        HTTPException: If the query is empty or the model name is not provided.
    """
    
    # Validate and process the request
    if not request.query or not request.model:
        raise HTTPException(status_code=400, detail="Query and model name are required")    
    
    # Strip whitespace from query and model
    user_query = request.query.strip()
    selected_model = request.model.strip() if request.model else None

    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not selected_model:
        raise HTTPException(status_code=400, detail="Model name is required")
    
    try:

        # Initialize the OllamaLLM with the specified model
        model= OllamaLLM(model=selected_model)

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

        # # Process query
        relevant_docs = retriever.invoke(user_query)
        information = "\n".join([doc.page_content for doc in relevant_docs])
        response = chain.invoke({"information": information, "query": user_query})

        # Return the response
        return response
        # Return the response in a structured format
        return {
            "query": user_query,
            "model": selected_model,
            "response": response
        }
    
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))