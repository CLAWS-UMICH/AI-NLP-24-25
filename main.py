# main.py

from langchain import OpenAI
from langchain.agents import initialize_agent
from tools import tools

llm = OpenAI(model_name="gpt-4o") 
agent = initialize_agent(tools=tools, llm=llm, agent_type="zero-shot-react-description")

# Define a function to interact with the agent
def langchain_agent_response(query: str):
    response = agent.run(query)
    return response

# Example usage
if __name__ == "__main__":
    print(langchain_agent_response("Example question"))
