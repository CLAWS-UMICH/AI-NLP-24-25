from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from tools import tools
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# for streaming response
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tools_used = []
    
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name")
        print(f"Tool used: {tool_name}")
        if tool_name:
            self.tools_used.append(tool_name)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end='', flush=True)

callback_handler = StreamingCallbackHandler()

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    verbose=False,
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

def get_tools_used():
    # Return a copy of the list to prevent external modifications
    return callback_handler.tools_used.copy()

def clear_tools_used():
    callback_handler.tools_used = []

def langchain_agent_response(query: str):
    clear_tools_used()  # Clear tools before each response
    response = agent.invoke(
        {"input": query},
        {"callbacks": [callback_handler]}
    )
    return response

# Example usage
if __name__ == "__main__":
    prompt = input("Astronaut prompt: ")
    langchain_agent_response(prompt)