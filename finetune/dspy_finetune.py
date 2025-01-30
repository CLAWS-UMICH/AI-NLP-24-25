import sys
sys.path.append('..')
from betterlangchain.tools import get_vitals, get_system_status
from dspy_formatted_data import training_data
import dspy

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField()

def get_vitals_tool(patient: str = None, *args, **kwargs) -> str:
    """Returns vital signs information"""
    return get_vitals()

def get_system_tool(patient: str = None, *args, **kwargs) -> str:
    """Returns system status information"""
    return get_system_status()

llm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=llm)

react_agent = dspy.ReAct(
    "question -> answer",  # Simple signature for question answering
    tools=[get_vitals_tool, get_system_tool],  # List of available tools
    max_iters=5  # Maximum number of reasoning/action steps
)

unlabeled_trainset = [data[0] for data in training_data]
print(react_agent(question=unlabeled_trainset[0]).answer)

t5_small = dspy.LM(model="huggingface/google-t5/t5-small")

rag_t5 = dspy.ReAct(
    "question -> answer",  # Simple signature for question answering
    tools=[get_vitals_tool, get_system_tool],  # List of available tools
    max_iters=5  # Maximum number of reasoning/action steps
)
# We chnage the language model (lm) for each predictor & use t5-small
for p in rag_t5.predictors():
    p.lm = t5_small
    
print(rag_t5(question=unlabeled_trainset[0]).answer)