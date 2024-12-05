import json
from typing import List, Dict
import pandas as pd
from datasets import Dataset

def load_training_data(file_path: str = "training_data.json") -> List[Dict]:
    """Load existing training data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_conversation_history(messages: List[Dict]) -> str:
    """Format conversation history into a single string."""
    history = []
    for msg in messages:
        if msg.get("role") in ["user", "assistant"]:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            history.append(f"{role}: {content}")
    return "\n".join(history)

def find_last_plan_message(conversations: List[Dict]) -> int:
    """Find the index of the last PLAN message in the conversation."""
    for i in reversed(range(len(conversations))):
        msg = conversations[i]
        if (msg.get("role") == "assistant" and 
            msg.get("content", "").strip().startswith("PLAN -")):
            return i
    return -1

def format_for_flan(data: List[Dict]) -> List[Dict]:
    """Format training data for FLAN-T5 fine-tuning."""
    examples = []
    
    for item in data:
        conversations = item.get('conversations', [])
        last_plan_idx = find_last_plan_message(conversations)
        
        if last_plan_idx > 0:  # If we found a plan message
            # Get all conversation history up to but not including the final plan
            history = format_conversation_history(conversations[:last_plan_idx])
            plan_text = conversations[last_plan_idx].get("content", "").strip()
            
            if history and plan_text:
                examples.append({
                    "sentence": history,
                    "text_label": plan_text  # Keep the full message including "PLAN -"
                })
    
    return examples

def create_flan_dataset(output_file: str = "flan_training_data.json"):
    """Create a dataset formatted for FLAN-T5 fine-tuning."""
    # Load existing training data
    training_data = load_training_data()
    
    # Format data for FLAN-T5
    training_examples = format_for_flan(training_data)
    
    # Convert to DataFrame and then to HuggingFace Dataset
    if training_examples:
        df = pd.DataFrame(training_examples)
        dataset = Dataset.from_pandas(df)
        
        # Save as JSON for inspection
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2, ensure_ascii=False)
        
        print(f"Processed training data into {len(training_examples)} examples")
        return dataset
    else:
        print("No valid training examples found!")
        return None

if __name__ == "__main__":
    dataset = create_flan_dataset()
    if dataset:
        print("\nDataset sample:")
        print(dataset[:2])