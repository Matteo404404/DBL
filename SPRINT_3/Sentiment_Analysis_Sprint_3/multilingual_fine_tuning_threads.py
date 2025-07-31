import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import transformers
from sklearn.preprocessing import LabelEncoder
import evaluate
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent


def fine_tune_multilang_sentiment():
    # config
    csv_path = base_dir / "CSVs"/ "random_threads_cumulative_cleaned_for_manual_labeling.csv"
    model_name = "FacebookAI/xlm-roberta-large"
    output_model_dir = "./xlm_roberta_thread_finetuned_multilang"
    max_seq_length = 512
    train_batch_size = 4
    eval_batch_size = 32
    learning_rate = 2e-5
    num_epochs = 3
    weight_decay = 0.01
    test_split_size = 0.2
    random_seed = 42

    # csv
    print(f"loading CSV from: {csv_path}")
    print("Attempting to load CSV by skipping the header and defining column names explicitly...")
    dataset_full = load_dataset("csv", data_files=str(csv_path), column_names=['text', 'label'], skiprows=1)

    # train/test
    print(f"Split 80/20 with seed {random_seed}...")
    split = dataset_full["train"].train_test_split(test_size=test_split_size, seed=random_seed)
    dataset = DatasetDict({"train": split["train"], "test": split["test"]})
    print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")


    print("encoding (negative, neutral, positive)...")
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset["train"]["label"])
    num_labels = len(label_encoder.classes_)

    def encode_labels(example):
        example["label"] = label_encoder.transform([example["label"]])[0]
        return example

    dataset = dataset.map(encode_labels)
    print(f"encoded labels: {label_encoder.classes_}")

    # tokenizer
    print(f"loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_seq_length)

    print("tokenization...")
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


    print(f"loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # evaluation
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]}


    training_args = TrainingArguments(
        output_dir=output_model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train and evaluate
    print("start training...")
    trainer.train()

    print("evaluation on test set...")
    eval_results = trainer.evaluate()
    print(f"\n Result test: {eval_results}")

    # save model
    print(f" Saving in: {output_model_dir}")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print("Fine-tuning complete.")


if __name__ == "__main__":
    os.makedirs("./xlm_roberta_finetuned_multilang", exist_ok=True)
    fine_tune_multilang_sentiment()
