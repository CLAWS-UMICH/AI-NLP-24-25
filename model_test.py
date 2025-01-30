from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

inputs = tokenizer(input("Enter a prompt: "), return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=600)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
