import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import evaluate

# config
fine_tuned_model = "Matteo404404/airline_tuned_roberta"
base_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
max_seq_length = 128
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load
dataset = load_dataset("osanseviero/twitter-airline-sentiment")["train"]
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

# encoding labels
label_encoder = LabelEncoder()
label_encoder.fit(dataset["train"]["airline_sentiment"])

def encode_labels(example):
    example["label"] = label_encoder.transform([example["airline_sentiment"]])[0]
    return example

test_dataset = test_dataset.map(encode_labels)

# load evaluation metric
accuracy = evaluate.load("accuracy")

# evaluation function
def evaluate_model(model_name):
    print(f"\nEvaluating: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_seq_length)

    tokenized = test_dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    all_preds = []
    all_labels = []

    loader = torch.utils.data.DataLoader(tokenized, batch_size=batch_size)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, axis=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return accuracy.compute(predictions=all_preds, references=all_labels)

# run
base_acc = evaluate_model(base_model)
fine_tuned_acc = evaluate_model(fine_tuned_model)


print("\n Accuracy Comparison ")
print(f"Base model ({base_model}): {base_acc['accuracy']:.4f}")
print(f"Fine-tuned model ({fine_tuned_model}): {fine_tuned_acc['accuracy']:.4f}")
