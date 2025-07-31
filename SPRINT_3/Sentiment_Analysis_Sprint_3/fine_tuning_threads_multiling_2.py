import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import evaluate
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent 

#  Metrics Function
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred):
    """Computes and returns a dictionary of metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "macro_f1": f1["f1"],
        "macro_precision": precision["precision"],
        "macro_recall": recall["recall"]
    }


def continue_fine_tuning():
    # config
    model_path = "./xlm_roberta_thread_finetuned_multilang"


    num_epochs = 7

    # Other parameters
    csv_path = base_dir / "CSVs"/ "random_threads_cumulative_cleaned_for_manual_labeling.csv"
    max_seq_length = 512
    train_batch_size = 4
    eval_batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.01
    random_seed = 42

    # Data Loading and
    print(f"Loading CSV from: {csv_path}")
    dataset_full = load_dataset("csv", data_files=csv_path, column_names=['text', 'label'], skiprows=1)

    print(f"Splitting data with seed {random_seed}...")
    split = dataset_full["train"].train_test_split(test_size=0.2, seed=random_seed)
    dataset = DatasetDict({"train": split["train"], "test": split["test"]})
    print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    # Fit on the full dataset labels
    all_labels = dataset['train']['label'] + dataset['test']['label']
    label_encoder.fit(all_labels)
    num_labels = len(label_encoder.classes_)

    def encode_labels(example):
        example["label"] = label_encoder.transform([example["label"]])[0]
        return example

    dataset = dataset.map(encode_labels)
    print(f"Encoded labels: {list(label_encoder.classes_)}")

    #load previously fine-tuned model
    print(f"Loading previously fine-tuned model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_seq_length)

    print("Tokenizing dataset...")
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # train
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
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

    # resume training from the last checkpoint
    print("Resuming training from the last checkpoint...")
    trainer.train(resume_from_checkpoint=True)

    print("Continued training complete.")

    # evaluate
    print("\n--- Final Evaluation on Test Set ---")
    final_eval_results = trainer.evaluate()
    print(final_eval_results)

    # classification and confusion matrix
    print("\n--- Detailed Classification Report ---")
    # Get predictions on the test set
    test_predictions = trainer.predict(dataset["test"])
    predicted_labels = np.argmax(test_predictions.predictions, axis=1)
    true_labels = dataset["test"]["label"]

    # Print detailed report
    print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()  # This will pop up as it is not in a notebook


if __name__ == "__main__":
    continue_fine_tuning()