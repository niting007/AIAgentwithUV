from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rich import print

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

def main():
    print("Hello from aiagentwithuv!")