from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from tools import tools
import os

llm = ChatOpenAI(model_name="gpt-4")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

def langchain_agent_response(query: str):
    response = agent.invoke(query)
    return response

# Example usage
if __name__ == "__main__":
    prompt = input("Astronaut prompt: ")
    print(langchain_agent_response(prompt))
