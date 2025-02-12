from better import Agent, Tool, Param
import random
import json
current_tasks = []

# List of diverse prompts for testing agent responses
test_prompts = [
   
]


def set_current_tasks(amount):
    for _ in range(amount):
        task = random.choice(potential_tasks)
        current_tasks.append(task)

def GetTasks():
    return current_tasks


# def main():
#     # Create agent with tools
#     agent = Agent([
#         Tool(
#             name="ClickTasks",
#             description="Opens the main task list interface.",
#             params=[],
#             return_description="True if successful, False otherwise",
#             function=ClickTasks
#         ),
#         Tool(
#             name="ClickNavigation", 

#             description="Opens the navigation interface.",
#             params=[],
#             return_description="True if successful, False otherwise",
#             function=ClickNavigation

#         ),
#         Tool(
#             name="ClickMessages", 
#             description="Opens the messages interface.",
#             params=[],
#             return_description="True if successful, False otherwise",
#             function=ClickMessages

#         ),
#         Tool(
#             name="ClickSamples", 
#             description="Opens the samples interface.",
#             params=[],
#             return_description="True if successful, False otherwise",
#             function=ClickSamples

#         ),
#         Tool(
#             name="ClickVitals", 
#             description="Opens the vitals interface.",
#             params=[],
#             return_description="True if successful, False otherwise",
#             function=ClickVitals

#         )
#     ])

#     print(f"Processing {len(test_prompts)} diverse prompts...")
    
#     for i, prompt in enumerate(test_prompts, 1):
#         print(f"\nProcessing prompt {i}/{len(test_prompts)}: {prompt}")
#         agent.clear_conversation_history()
#         response = agent.ask(prompt)
#         print(f"Response received and saved ------------------------------------------------------------------------")

# if __name__ == "__main__":
#     main()
