import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import evaluate

def fine_tune_airline_sentiment():
    # config
    dataset_name = "osanseviero/twitter-airline-sentiment"

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    output_model_dir = "./airline_tuned_roberta_best"  # Will be created in the same directory
    max_seq_length = 128  # Max length for tokenized tweets
    train_batch_size = 16
    eval_batch_size = 64
    learning_rate = 2e-5
    num_epochs = 3 # Keep low for now
    weight_decay = 0.01
    test_split_size = 0.2 # of the data for evaluation
    random_seed = 42

    # prepare dataset
    print(f"Loading dataset: {dataset_name}...")
    try:
        dataset_full = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection and the dataset name is correct.")
        return

    print(f"Original dataset splits: {list(dataset_full.keys())}")

    # Check if 'train' split exists
    if 'train' not in dataset_full:
        print(f"Error: The dataset '{dataset_name}' does not contain a 'train' split.")
        print("Please check the dataset structure on Hugging Face Hub.")
        return

    # Split the 'train' data into train and test
    print(f"Splitting 'train' data into train/test ({1-test_split_size:.0%}/{test_split_size:.0%})...")
    train_test_split_data = dataset_full['train'].train_test_split(test_size=test_split_size, seed=random_seed)

    # Create a new DatasetDict
    dataset = DatasetDict({
        'train': train_test_split_data['train'],
        'test': train_test_split_data['test']
    })
    print(f"New dataset splits: {list(dataset.keys())}")
    print(f"Train samples: {len(dataset['train'])}, Test samples: {len(dataset['test'])}")

    # encoding labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset['train']['airline_sentiment'])
    num_labels = len(label_encoder.classes_)
    print(f"Found {num_labels} unique labels: {label_encoder.classes_}")

    def encode_labels_fn(example):
        example["label"] = label_encoder.transform([example["airline_sentiment"]])[0]
        return example

    dataset = dataset.map(encode_labels_fn, batched=False)
    print("Labels encoded.")

    # tokenization
    print(f"Loading tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_seq_length)

    print(f"Tokenizing dataset (max_length: {max_seq_length})...")
    dataset = dataset.map(tokenize_batch, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print("Dataset tokenized and formatted for PyTorch.")

    # load model
    print(f"Loading model: {model_name} for {num_labels} labels...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    # training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=2,
        report_to="none"
    )

    # trainer setup
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        # argmax.
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn
    )

    # actually training
    print("Starting training...")
    try:
        trainer.train()
        print("Training finished.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

    # evaluation
    print("Evaluating the best model on the test set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    #save model and tokenizer
    # The Trainer with load_best_model_at_end=True will have the best model loaded.

    print(f"Saving best model and tokenizer to {output_model_dir}...")
    trainer.save_model(output_model_dir)

    print(f"Fine-tuning complete. Model and tokenizer saved to {output_model_dir}")

    # checking the saved model with an example to see if it works
    print("\n--- Example of loading and using the saved model ---")
    try:
        loaded_model = AutoModelForSequenceClassification.from_pretrained(output_model_dir)
        loaded_tokenizer = AutoTokenizer.from_pretrained(output_model_dir)
        loaded_model.to(device)

        sample_text = "This airline is fantastic! Great service."
        print(f"Sample text: {sample_text}")

        inputs = loaded_tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_length).to(device)
        with torch.no_grad():
            logits = loaded_model(**inputs).logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
        print(f"Predicted sentiment: {predicted_label} (ID: {predicted_class_id})")

        sample_text_neg = "Absolutely terrible flight, worst experience ever."
        print(f"Sample text: {sample_text_neg}")
        inputs_neg = loaded_tokenizer(sample_text_neg, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_length).to(device)
        with torch.no_grad():
            logits_neg = loaded_model(**inputs_neg).logits
        predicted_class_id_neg = torch.argmax(logits_neg, dim=1).item()
        predicted_label_neg = label_encoder.inverse_transform([predicted_class_id_neg])[0]
        print(f"Predicted sentiment: {predicted_label_neg} (ID: {predicted_class_id_neg})")

    except Exception as e:
        print(f"Error during example usage of saved model: {e}")

if __name__ == "__main__":
    output_dir_path = "./airline_tuned_roberta_best"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)

    fine_tune_airline_sentiment()