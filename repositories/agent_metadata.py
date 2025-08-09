from rich import print
import os
from dotenv import load_dotenv
import requests

load_dotenv()

def getModels():
    """
    Fetches the list of models currently installed locally using the Ollama API.
    
    Returns:
        list: A list of model names currently available.
    """
    models = []
    try:
        OLLAMA_SERVER_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
        # Read skip list from env (comma-separated)
        skip_list = set(m.strip() for m in os.getenv("SKIP_MODELS", "").split(",") if m.strip())

        resp = requests.get(f"{OLLAMA_SERVER_URL}/api/tags")
        data = resp.json()
        print(f"[bold green]Models available: {data.get('models', [])}[/bold green]")
        
        # Filter models: remove version tags and skip those in skip_list
        models = [
            m["model"].split(":")[0]
            for m in data.get("models", [])
            if m["model"].split(":")[0] not in skip_list
        ]
        
        print(models)

        return models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return models
    

    