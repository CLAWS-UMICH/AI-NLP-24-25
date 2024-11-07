import json
import pandas as pd

output_file = 'data_generation/output.jsonl'
generated_file = 'data_generation/generated_data1.jsonl'

data = []

with open(output_file, 'r') as f1, open(generated_file, 'r') as f2:

    for response, prompt in zip(f1.readlines(), f2.readlines()):

        response = json.loads(response)
        prompt = json.loads(prompt)

        content = response['response']['body']['choices'][0]['message']['content']
        bucket = prompt['body']['messages'][1]['content'].split()[19]

        data.append({'content': content, 'bucket': bucket})

df = pd.DataFrame(data)
df = df.drop_duplicates(subset=['content'])
print(df)

df.to_csv('data_generation/data.csv', index=False)   
