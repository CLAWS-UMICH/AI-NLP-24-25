from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackHandler
from tools import tools
import os

# for streaming response
class StreamingCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end='', flush=True)

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingCallbackHandler()],
    verbose=False
)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

def langchain_agent_response(query: str):
    response = agent.run(query)
    return response

# Example usage
if __name__ == "__main__":
    prompt = input("Astronaut prompt: ")
    langchain_agent_response(prompt)
