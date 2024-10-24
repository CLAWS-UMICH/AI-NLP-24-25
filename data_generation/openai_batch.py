import os
import json
from dotenv import load_dotenv
from random import randint, seed
from openai import OpenAI
from copy import deepcopy

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PROMPT = f"""Generate a simple voice command from an astronaut that would be classified into the SELECTED_BUCKET category. It is mostly tied to pulling up screens on their helmet. For example, Open navigation panel or see my messages"""


# Set your OpenAI API key here

seed(42)

def generate_one_data_point(buckets, model="gpt-4o-mini"):
    """
    Generates a dataset of sentences and their classifications into given buckets.

    Parameters:
    - num_samples: Number of samples to generate.
    - buckets: List of bucket names for classification.
    - model: The OpenAI model used for generating sentences and classifying.

    Returns:
    - List of dictionaries containing generated sentences and their assigned bucket.
    """
    datapoints = []
    selected_bucket = buckets[randint(0, len(buckets))]


    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data generator"},
                {
                    "role": "user",
                    "content": PROMPT.replace("SELECTED_BUCKET", selected_bucket)
                }
            ]
        )

        generated_sentence = response.choices[0].message.content.strip().replace('"', '')

        datapoint = {
            "sentence": generated_sentence,
            "bucket": selected_bucket
        }
        datapoints.append(datapoint)
    
    except Exception as e:
        print(f"An error occurred while generating the data point: {e}")

    return datapoints

def generate_jsonl_batch_request(num_samples, buckets, model="gpt-4o-mini", file_name="generated_data.jsonl"):
    template = {"custom_id": "request-", "method": "POST", "url": "/v1/chat/completions", "body": {"model": model, "messages": [{"role": "system", "content": "You are a helpful data generator"}, {"role": "user", "content": PROMPT}],"max_tokens": 100}}
    open(file_name, "w").close() #remove everything
    with open(file_name, "a") as f:
        for i in range(num_samples):
            selected_bucket = buckets[randint(0, len(buckets))]
            new_batch = deepcopy(template)
            new_batch['custom_id'] += str(i)
            new_batch['body']['messages'][1]['content'] = new_batch['body']['messages'][1]['content'].replace("SELECTED_BUCKET", selected_bucket)
    
            f.write(json.dumps(new_batch) + '\n')

def main():
    # Define the categories (buckets)
    buckets = ["Navigation", "Geosampling", "Vitals", "Samples", "UIA", "Messages"]

    # Number of data samples to generate
    num_samples = 1  # You can adjust this as needed

    # Generate the data points
    # generated_data = generate_one_data_point(num_samples, buckets)
    
    # print(generated_data)
    generate_jsonl_batch_request(10, buckets)

if __name__ == "__main__":
    main()
