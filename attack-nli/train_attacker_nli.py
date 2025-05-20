import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from copy import deepcopy
import random
import argparse
import datetime
timestamp = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
import json
import requests
import hashlib
import diskcache
cache = diskcache.Cache('/usa/taikun/07_transencoder/attack-genai', size_limit=10e9)
cache.stats(enable=True)    
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import train_test_split
import GPUtil
from tqdm.auto import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    AutoModelForSequenceClassification,
    BertConfig,
    DataCollatorWithPadding
)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import DataLoader, Subset, RandomSampler
torch.set_default_dtype(torch.float32)
# torch.cuda.set_device(0)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
device = 'cpu'
print(f"Using GPU {device}")

import tensorflow as tf
import tensorflow_hub as hub
tf.config.set_visible_devices([], 'GPU')

def get_raw_logits_nli(data:list, tgt_model):
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(item["prompt"])
        else:
            prompts.append(json.dumps(item))
    
    # Process each prompt
    results = []
    for i, prompt in enumerate(prompts):
        # print(f"Processing prompt {i+1}/{len(prompts)}")
        result = get_raw_logits(prompt, server_url, model_name)
        results.append(result)
    
    prompt, predictions, probs = [], [], []
    for result in results:
        prompt.append(result['prompt'])
        predictions.append(0 if result['all_logprobs']['token_1']['token'] == 'safe' else 1)
        probs.append(np.exp(result['all_logprobs']['token_1']['logprob']))

    return prompt, predictions, probs

def apply_random_masks(input_ids, attention_mask, num_masks=10, mask_token_id=None):
    """
    Randomly select and mask tokens in the input_ids tensor.
    Returns the masked input_ids and the positions of the masks.
    """
    batch_size = input_ids.size(0)
    seq_length = input_ids.size(1)
    
    # Create copies to modify
    masked_input_ids = input_ids.clone()
    mask_positions = []
    
    for i in range(batch_size):
        # Find valid positions (non-padding tokens)
        valid_positions = torch.nonzero(attention_mask[i] == 1).squeeze().tolist()
        if isinstance(valid_positions, int):  # Handle case with only one valid position
            valid_positions = [valid_positions]
        
        # Skip if there aren't enough valid positions
        if len(valid_positions) <= num_masks:
            mask_positions.append([])
            continue
            
        # Choose random positions to mask
        chosen_positions = random.sample(valid_positions, num_masks)
        mask_positions.append(chosen_positions)
        
        # Apply masks
        for pos in chosen_positions:
            masked_input_ids[i, pos] = mask_token_id
    
    return masked_input_ids, mask_positions

def logits_to_labels_doc(input_ids, masked_positions, token_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id):
    batch_size = input_ids.size(0)
    vocab_size = token_logits.size(-1)
    
    # Initialize storage for generated tokens
    generated_tokens = []
    for i in range(batch_size):
        positions = masked_positions[i]
        batch_generated = []
        
        for j, pos in enumerate(positions):
            # Get logits for this specific masked position
            pos_logits = token_logits[i, pos]
            
            # Sample a token based on the probabilities
            pos_probabilities = F.softmax(pos_logits, dim=-1)
            sampled_token = torch.multinomial(pos_probabilities.view(-1), num_samples=1).item()
            batch_generated.append((pos, sampled_token))
        
        generated_tokens.append(batch_generated)
    
    # Create modified documents
    original_input_ids_list = input_ids.tolist()
    source_documents = []
    generated_documents = []
    generated_labels = []
    
    for i in range(batch_size):
        # Get original document
        original_document_tokens = [
            token_id for token_id in original_input_ids_list[i] 
            if token_id != pad_token_id and token_id not in [cls_token_id, sep_token_id]
        ]
        original_document = tokenizer.decode(original_document_tokens, skip_special_tokens=True)
        source_documents.append(original_document)
        
        # Create modified document
        modified_tokens = original_input_ids_list[i].copy()
        for pos, new_token in generated_tokens[i]:
            modified_tokens[pos] = new_token
            
        # Filter out padding and special tokens
        modified_tokens = [
            token_id for token_id in modified_tokens
            if token_id != pad_token_id and token_id not in [cls_token_id, sep_token_id]
        ]
        
        modified_document = tokenizer.decode(modified_tokens, skip_special_tokens=True)
        generated_documents.append(modified_document)
        
        # Store generated token values for loss calculation
        token_values = [new_token for _, new_token in generated_tokens[i]]
        generated_labels.append(token_values)
    
    return batch_size, vocab_size, generated_labels, source_documents, generated_documents

def load_and_prepare_data(data_name, split, max_len=512):
    """
    Loads the specified dataset and prepares the tokenizers and label mappings.
    Returns the dataset and relevant tokenizers and mappings.

    Args:
        data_name (str): Name of the dataset to load (e.g., 'snli').
        split (str): The dataset split to load (e.g., 'train', 'validation').
        max_len (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        num_examples (int, optional): Number of examples to load for demonstration. Defaults to 5.

    Returns:
        tuple: (dataset, atk_tokenizer, tgt_tokenizer, ds_to_model)
            - dataset (datasets.Dataset): The loaded dataset.
            - ds_to_model (dict): Mapping from dataset label indices to model label IDs.
    """
    # model_label2id = {v: k for k, v in tgt_model.config.id2label.items()}
    model_label2id = {"contradiction": 0, "entailment": 1, "neutral": 2,}

    if data_name == "snli":
        dataset = load_dataset("stanfordnlp/snli", split=split)
    else:
        raise ValueError(f"Unsupported data_name: {data_name}")
    
    dataset = dataset.filter(lambda example: example["label"] != -1)
    dataset_label_names = dataset.features["label"].names
    ds_to_model = {
        ds_idx: model_label2id[name] 
        for ds_idx, name in enumerate(dataset_label_names) if name in model_label2id
    }
    print(f"Dataset→Model ID map for {split}: {ds_to_model}")

    return dataset, ds_to_model

def tokenize_dataset(dataset, tokenizer, ds_to_model, max_length=512):
    """
    Tokenizes the dataset and maps dataset labels to model labels.
    
    Args:
        dataset: The dataset to tokenize
        tokenizer: The tokenizer to use
        ds_to_model: Mapping from dataset label indices to model label IDs
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset with input_ids, attention_mask, and labels
    """
    def preprocess_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples['premise'],
            examples['hypothesis'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        
        # Map dataset labels to model labels
        tokenized['labels'] = [ds_to_model.get(label) for label in examples['label']]
        
        return tokenized
    
    # Apply tokenization to the entire dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    return tokenized_dataset

def build_dataloaders_limited_train(atker_path, target_path, data_name, train_limit=10000, batch_size=16, shuffle_train=True, eval_batch_size=1, seed=42):
    """
    Build dataloaders with a limited number of training samples.
    
    Args:
        atker_path: Path to the attacker model
        target_path: Path to the target model
        data_name: Name of the dataset
        train_limit: Maximum number of training samples to use
        batch_size: Batch size for training and validation
        shuffle_train: Whether to shuffle the training data
        eval_batch_size: Batch size for evaluation
        seed: Random seed
        
    Returns:
        train_dataloader, validation_dataloader, evaluation_dataloader, train_atk_tokenizer
    """
    # Load the datasets
    print("Loading training data...")
    atk_tokenizer = AutoTokenizer.from_pretrained(atker_path)
    train_dataset, train_label_map = load_and_prepare_data(data_name, split="train")
    
    # Limit the training dataset
    if len(train_dataset) > train_limit:
        train_dataset = train_dataset.select(range(train_limit))
        print(f"Training data limited to {train_limit} samples.")
    
    print(f"Training data loaded. Number of examples: {len(train_dataset)}")

    print("\nLoading validation data...")
    val_dataset, val_label_map = load_and_prepare_data(data_name, split="validation")
    print(f"Validation data loaded. Number of examples: {len(val_dataset)}")
    
    # Optional: Load test/evaluation data if available
    print("\nLoading evaluation data...")
    try:
        eval_dataset, eval_label_map = load_and_prepare_data(data_name, split="test")
        print(f"Evaluation data loaded. Number of examples: {len(eval_dataset)}")
    except:
        print("No separate evaluation dataset found. Using validation dataset for evaluation.")
        eval_dataset = val_dataset
        eval_label_map = val_label_map
    
    # Tokenize the datasets
    print("\nTokenizing training dataset...")
    tokenized_train = tokenize_dataset(train_dataset, atk_tokenizer, train_label_map)
    
    print("Tokenizing validation dataset...")
    tokenized_val = tokenize_dataset(val_dataset, atk_tokenizer, val_label_map)
    
    print("Tokenizing evaluation dataset...")
    tokenized_eval = tokenize_dataset(eval_dataset, atk_tokenizer, eval_label_map)
    
    # Set the format to PyTorch tensors
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create DataLoaders
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=shuffle_train)
    validation_dataloader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False)
    evaluation_dataloader = DataLoader(tokenized_eval, batch_size=eval_batch_size, shuffle=False)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Training data size: {len(tokenized_train)}")
    print(f"Validation data size: {len(tokenized_val)}")
    print(f"Evaluation data size: {len(tokenized_eval)}")
    
    print(f"\nNumber of batches in training dataloader: {len(train_dataloader)}")
    print(f"Number of batches in validation dataloader: {len(validation_dataloader)}")
    print(f"Number of batches in evaluation dataloader: {len(evaluation_dataloader)}")
    
    # If you want to print class distribution like in your example
    if hasattr(tokenized_train, 'features') and 'labels' in tokenized_train.features:
        print("\nClass distribution in each split:")
        print(f"Training: {get_class_distribution(tokenized_train)}")
        print(f"Validation: {get_class_distribution(tokenized_val)}")
        print(f"Evaluation: {get_class_distribution(tokenized_eval)}")
    
    return train_dataloader, validation_dataloader, evaluation_dataloader

def build_dataloaders(atker_path, target_path, data_name, batch_size=16, shuffle_train=True, eval_batch_size=1, seed=42):
    # Load the datasets
    print("Loading training data...")
    # train_dataset, train_atk_tokenizer, train_tgt_tokenizer, train_label_map = load_and_prepare_data(
    #     atker_path, target_path, data_name, split="train")
    train_dataloader, validation_dataloader, evaluation_dataloader = build_dataloaders_limited_train(
        atker_path=atker_path, 
        target_path=target_path, 
        data_name=data_name,
        train_limit=10000  # This limits your training set to 10k samples
    )
    # print(f"Training data loaded. Number of examples: {len(train_dataset)}")

    # print("\nLoading validation data...")
    # val_dataset, val_atk_tokenizer, val_tgt_tokenizer, val_label_map = load_and_prepare_data(
    #     atker_path, target_path, data_name, split="validation")
    # print(f"Validation data loaded. Number of examples: {len(val_dataset)}")
    
    # # Optional: Load test/evaluation data if available
    # print("\nLoading evaluation data...")
    # try:
    #     eval_dataset, eval_atk_tokenizer, eval_tgt_tokenizer, eval_label_map = load_and_prepare_data(
    #         atker_path, target_path, data_name, split="test")
    #     print(f"Evaluation data loaded. Number of examples: {len(eval_dataset)}")
    # except:
    #     print("No separate evaluation dataset found. Using validation dataset for evaluation.")
    #     eval_dataset = val_dataset
    #     eval_atk_tokenizer = val_atk_tokenizer
    #     eval_tgt_tokenizer = val_tgt_tokenizer
    #     eval_label_map = val_label_map
    
    # # Tokenize the datasets
    # print("\nTokenizing training dataset...")
    # tokenized_train = tokenize_dataset(train_dataset, train_atk_tokenizer, train_label_map)
    
    # print("Tokenizing validation dataset...")
    # tokenized_val = tokenize_dataset(val_dataset, val_atk_tokenizer, val_label_map)
    
    # print("Tokenizing evaluation dataset...")
    # tokenized_eval = tokenize_dataset(eval_dataset, eval_atk_tokenizer, eval_label_map)
    
    # # Set the format to PyTorch tensors
    # tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # # Create DataLoaders
    # train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=shuffle_train)
    # validation_dataloader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False)
    # evaluation_dataloader = DataLoader(tokenized_eval, batch_size=eval_batch_size, shuffle=False)
    
    # # Print statistics
    # print("\nDataset statistics:")
    # print(f"Training data size: {len(tokenized_train)}")
    # print(f"Validation data size: {len(tokenized_val)}")
    # print(f"Evaluation data size: {len(tokenized_eval)}")
    
    # print(f"\nNumber of batches in training dataloader: {len(train_dataloader)}")
    # print(f"Number of batches in validation dataloader: {len(validation_dataloader)}")
    # print(f"Number of batches in evaluation dataloader: {len(evaluation_dataloader)}")
    
    # # If you want to print class distribution like in your example
    # if hasattr(tokenized_train, 'features') and 'labels' in tokenized_train.features:
    #     print("\nClass distribution in each split:")
    #     print(f"Training: {get_class_distribution(tokenized_train)}")
    #     print(f"Validation: {get_class_distribution(tokenized_val)}")
    #     print(f"Evaluation: {get_class_distribution(tokenized_eval)}")
    
    return train_dataloader, validation_dataloader, evaluation_dataloader

def get_class_distribution(dataset):
    """
    Calculate the distribution of classes in a dataset.
    
    Args:
        dataset: A dataset with a 'labels' column
        
    Returns:
        A dictionary mapping class labels to their counts
    """
    if not hasattr(dataset, 'features') or 'labels' not in dataset.features:
        return "Labels not found in dataset"
    
    labels = dataset['labels']
    
    # Convert list to numpy array if it's a list
    if isinstance(labels, list):
        labels = np.array(labels)
    # If it's already a tensor, convert to numpy
    elif hasattr(labels, 'numpy'):
        labels = labels.numpy()
    
    # Handle NaN values separately
    nan_count = np.isnan(labels).sum()
    
    # Get unique non-NaN labels
    unique_labels = set(label for label in labels.flatten() if not np.isnan(label))
    
    # Count occurrences of each label
    distribution = {label: (labels == label).sum() for label in unique_labels}
    
    # Add NaN count if any exist
    if nan_count > 0:
        distribution[float('nan')] = nan_count
    
    # Calculate percentages
    total = sum(distribution.values())
    distribution_percent = {label: f"{int(count)} ({count/total:.1%})" 
                           for label, count in distribution.items()}
    
    return distribution_percent
    

def main(args):
    """
    Main function to load and display the SNLI dataset.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    atker_path = args.atker_path
    target_path = args.target_path
    data_name = args.data_name
    len_doc_max = args.len_doc_max
    num_doc_masks = args.num_doc_masks
    
    
    train_dataloader, validation_dataloader, evaluation_dataloader = build_dataloaders(
        atker_path=atker_path, 
        target_path=target_path, 
        data_name=data_name)
    
    USE = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")
    
    # Initialize model
    config = BertConfig.from_pretrained(atker_path, output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained(atker_path, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(atker_path, max_length=len_doc_max)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id

    # Freeze all layers except the MLM head
    for name, param in model.named_parameters():
        if 'cls' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    def train(model, train_dataloader, validation_dataloader, tokenizer, epochs=10, eval_interval=100, print_length=200):  
        model.train()
        best_acc, val_accuracy = float('inf'), float('inf')
        step = -1

        total_batches = len(train_dataloader)
        total_steps = epochs * total_batches

        bar = tqdm(total=total_steps, desc="Training", unit="step")
        source_documents_all, generated_documents_all, labels_all = [], [], []
        val_source_documents_all, val_generated_documents_all, val_labels_all = [], [], []
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_dataloader, start=1):
                step += 1
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                masked_input_ids, mask_positions = apply_random_masks(
                    input_ids, attention_mask, num_masks=num_doc_masks, mask_token_id=mask_token_id
                )
                logits = model(masked_input_ids, attention_mask).logits
                batch_size, vocab_size, generated_labels, source_documents, generated_documents = logits_to_labels_doc(
                    input_ids, mask_positions, logits, tokenizer, pad_token_id, cls_token_id, sep_token_id
                )
                
                adv_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
                sem_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
                rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
                _, predictions_, probs_ = get_raw_logits.process_file(data=generated_documents, server_url=server_url)
                labels_all.extend(predictions_)
                predicted_classes = torch.tensor(predictions_).to(device)
                
                USE_res = getUSEcosSimilarity(source_documents, generated_documents, embed=USE)
                for i, gen_doc in enumerate(generated_documents):                        
                    if predictions_[i] not in (1, 0):
                        if labels[i] == 0:
                            adv_rewards[i] = 0.0
                        else:
                            adv_rewards[i] = 1.0
                    elif labels[i] == predictions_[i]:
                        adv_rewards[i] = 1 - probs_[i]
                    else:
                        adv_rewards[i] = probs_[i]

                    rewards[i] = alpha * adv_rewards[i] + (1-alpha) * sem_rewards[i]
                
                # 5. Calculate loss for document-based approach
                loss = torch.tensor(0.0, device=device)
                for i in range(batch_size):
                    # Skip samples with no masked positions
                    if not mask_positions[i]:
                        continue
                        
                    batch_losses = []
                    for j, pos in enumerate(mask_positions[i]):
                        token_logits = logits[i, pos]
                        token_label = torch.tensor(generated_labels[i][j], device=device)
                        token_loss = F.cross_entropy(token_logits.unsqueeze(0), token_label.unsqueeze(0))
                        batch_losses.append(token_loss)
                        
                    if batch_losses:
                        batch_loss = torch.mean(torch.stack(batch_losses))
                        loss += batch_loss * rewards[i]

    train(model=model, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, tokenizer=tokenizer)  
                
                

    # print("Loading training data...")
    # train_dataset, train_atk_tokenizer, train_tgt_tokenizer, train_label_map = load_and_prepare_data(
    #     atker_path, target_path, data_name, split="train")
    # print(f"Training data loaded. Number of examples: {len(train_dataset)}")

    # print("\nLoading validation data...")
    # val_dataset, val_atk_tokenizer, val_tgt_tokenizer, val_label_map = load_and_prepare_data(
    #     atker_path, target_path, data_name, split="validation")
    # print(f"Validation data loaded. Number of examples: {len(val_dataset)}")

    # # Example of accessing the loaded data
    # print("\nExample of accessing data:")
    # for i in range(min(2, len(train_dataset))):
    #     example = train_dataset[i]
    #     premise = example['premise']
    #     hypothesis = example['hypothesis']
    #     label_index = example['label']
    #     label_name = train_dataset.features["label"].names[label_index]
    #     model_label_id = train_label_map.get(label_index)
    #     print(f"--- Example {i + 1} (Train) ---")
    #     print(f"Premise: {premise}")
    #     print(f"Hypothesis: {hypothesis}")
    #     print(f"Original Label Index: {label_index}, Name: {label_name}")
    #     print(f"Mapped Model Label ID: {model_label_id}")

    # for i in range(min(2, len(val_dataset))):
    #     example = val_dataset[i]
    #     premise = example['premise']
    #     hypothesis = example['hypothesis']
    #     label_index = example['label']
    #     label_name = val_dataset.features["label"].names[label_index]
    #     model_label_id = val_label_map.get(label_index)
    #     print(f"\n--- Example {i + 1} (Validation) ---")
    #     print(f"Premise: {premise}")
    #     print(f"Hypothesis: {hypothesis}")
    #     print(f"Original Label Index: {label_index}, Name: {label_name}")
    #     print(f"Mapped Model Label ID: {model_label_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and prepare the SNLI dataset.")
    parser.add_argument("--atker_path", type=str, default="bert-base-uncased",
                        help="Path to the attacker model tokenizer.")
    parser.add_argument("--target_path", type=str, default="textattack/bert-base-uncased-snli",
                        help="Path to the target model.")
    parser.add_argument("--data_name", type=str, default="snli",
                        help="Name of the dataset to load ('snli' or 'mnli').")
    parser.add_argument("--len_doc_max", type=int, default=512)
    # args = parser.parse_args()

    args = argparse.Namespace(
        atker_path='bert-base-uncased', # Example path
        target_path='temp',
        data_name='snli',
        len_doc_max=256,
        num_doc_masks=10)

    main(args)
