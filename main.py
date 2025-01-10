import json
from preprocess import load_spider_dataset, preprocess_dataset
from model import initialize_model, tokenize_dataset, fine_tune_model
from search_and_learn import search_and_learn

def main():
    dataset = load_spider_dataset()
    train_data, val_data = preprocess_dataset(dataset)

    tokenizer, model = initialize_model()
    tokenized_train = tokenize_dataset(train_data, tokenizer)
    tokenized_val = tokenize_dataset(val_data, tokenizer)
    
    fine_tune_model(model, tokenizer, tokenized_train, tokenized_val)
    
    results = []
    for example in val_data[:10]:  
        result = search_and_learn(example, model, tokenizer)
        results.append(result)

    with open("human_eval.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved to human_eval.json")
    
if __name__ == "__main__":
    main()
