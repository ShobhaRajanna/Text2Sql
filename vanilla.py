import json
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

dataset = load_dataset("xlangai/spider2-lite")
train_data = dataset["train"]
print(f'length of train_data: {len(train_data)}')
model_name = "t5-small"  
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  
model = T5ForConditionalGeneration.from_pretrained(model_name)

def test_model_on_spider_lite(example):

    if not isinstance(example, dict):
        print("Error: Example is not a dictionary:", example)
        return None

    question = example["question"]  
    schema = example["db"] 

    input_text = f"Generate SQL: {question} | Schema: {schema}"

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "question": question,
        "schema": schema,
        "generated_sql": generated_sql,
    }
unique_schemas = set(example["db"] for example in train_data)
num_unique_schemas = len(unique_schemas)
print(f"Number of unique schemas in train_data: {num_unique_schemas}")
print(f"Unique schemas: {unique_schemas}")
#save in json format
output_file = "unique_schemas.json"
with open(output_file, "w") as f:
    json.dump(list(unique_schemas), f, indent=4)
        
exit()
results = []
for i, example in enumerate(train_data):  
    if i >= 5:  
        break
    print(f"Processing example {i+1}")
    result = test_model_on_spider_lite(example)
    if result:
        results.append(result)
        print(json.dumps(result, indent=4))  

output_file = "vanilla_t5_spider_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
