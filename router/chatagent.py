from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import repositories.agent_communication
import repositories.agent_metadata
router = APIRouter()

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
        agentReply = repositories.agent_communication.sendMessageToModle(modelName=selected_model, message=user_query)
        print(f"User Query: {user_query}")
        print(f"Selected Model: {selected_model}")
        print("Agent Response:")
        print(agentReply["response"])
        return agentReply["response"]
    
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))