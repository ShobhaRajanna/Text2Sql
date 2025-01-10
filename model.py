from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

def initialize_model(model_name="t5-small"):

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def tokenize_dataset(dataset, tokenizer):
 
    def tokenize_function(example):
        inputs = tokenizer(example["input_text"], max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(example["output_text"], max_length=512, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    return dataset.map(tokenize_function, batched=True)

def fine_tune_model(model, tokenizer, train_data, val_data):

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer
