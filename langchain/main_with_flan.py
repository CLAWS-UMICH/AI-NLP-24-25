from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from tools import tools

prompt_template = PromptTemplate(
    input_variables=["query", "agent_scratchpad", "tool_names", "tools"],
    template=(
        "You are an astronaut assistant. Answer the following query in a detailed and structured manner:\n"
        "Query: {query}\n"
        "Tools available: {tool_names}\n"
        "Tools details: {tools}\n"
        "Agent scratchpad: {agent_scratchpad}\n"
        "Provide your response in a clear and organized format."
    )
)

# Load Flan-T5 Small
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a HuggingFace pipeline with increased max_new_tokens
hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2560,  # Increase this value as needed
)

# Use LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Initialize the agent
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt_template,
)

def langchain_agent_response(query: str):
    response = agent.invoke({
        "query": query,
        "intermediate_steps": [],
    })
    return response

# Example usage
if __name__ == "__main__":
    query = input("Astronaut prompt: ")
    print(langchain_agent_response(query))