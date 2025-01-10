from datasets import load_dataset

def load_spider_dataset():
    dataset = load_dataset("xlangai/spider2-lite")
    return dataset

def preprocess_dataset(dataset):

    def preprocess_example(example):
        input_text = (
            f"Generate SQL: {example['question']} | "
            f"Schema: {example['db']} | "
            f"Knowledge: {example['external_knowledge']} | "
            f"Temporal: {example['temporal']}"
        )
        output_text = example.get("sql_query", "")  
        return {"input_text": input_text, "output_text": output_text}

    train_data = dataset["train"].map(preprocess_example)
    val_data = dataset["validation"].map(preprocess_example)
    return train_data, val_data
