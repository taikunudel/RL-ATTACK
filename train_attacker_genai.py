#!/usr/bin/env python
# coding: utf-8

# ## 1. Prepare Dataloader
# Load model directly
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
import functools
import hashlib
from cache_to_disk import cache_to_disk
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

import tensorflow as tf
import tensorflow_hub as hub
tf.config.set_visible_devices([], 'GPU')
# # Your TensorFlow code here
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices)) # This should ideally print 1
# if len(physical_devices) > 0:
#     print("Using GPU:", physical_devices[0])

# from utils import *
import get_raw_logits

total_calls = 0
cache_hits = 0

def print_config(args):
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def getUSEcosSimilarity(srcDocs, copyDocs, embed):
    USEcosinSimilarity = []
    sim_metric = torch.nn.CosineSimilarity(dim=1)
    for src, copy in zip(srcDocs, copyDocs):
        emb1, emb2 = embed([src, copy])["outputs"]
        emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
        srcEmb = torch.unsqueeze(emb1, dim=0) # [embSz] -> [1, embSz]
        advEmb = torch.unsqueeze(emb2, dim=0)
        es = sim_metric(srcEmb, advEmb)
        USEcosinSimilarity.append(es.item())
    return USEcosinSimilarity

def cache_report(reset: bool = False) -> None:
    hits, misses = cache.stats(enable=False, reset=reset)
    total = hits + misses
    hr = hits / total if total else 0.0
    print(f"[Cache] hits={hits}  misses={misses}  hit‑rate={hr:.1%}")
    
def preprocess_function(examples, prefix_length, len_doc_max, tokenizer, atk_what='prefix'):
    """
    Preprocess function that handles different attack types:
    - 'prefix': Add [MASK] tokens as a prefix (original functionality)
    - 'doc': Keep original text but insert placeholders for later masking within document
    """
    if atk_what == 'prefix':
        # Original prefix-based approach
        mask_token = '[MASK]'
        prefix = ' '.join([mask_token] * prefix_length)
        inputs = [prefix + ' ' + doc for doc in examples['prompt']]
        tokenized_inputs = tokenizer(inputs, max_length=len_doc_max, truncation=True, padding='max_length')
    elif atk_what == 'doc':
        # New approach: we'll tokenize documents as-is
        # We'll mask tokens during training, not during preprocessing
        inputs = examples['prompt']
        tokenized_inputs = tokenizer(inputs, max_length=len_doc_max, truncation=True, padding='max_length')
    else:
        raise ValueError(f"Unknown attack type: {atk_what}")
        
    tokenized_inputs['labels'] = [1 if j else 0 for j in examples['jailbreak']]
    return tokenized_inputs

def get_class_distribution(dataset):
    jailbreak_count = sum(dataset['labels'])
    regular_count = len(dataset) - jailbreak_count
    return {"jailbreak": jailbreak_count, "regular": regular_count, "total": len(dataset)}

def build_datasets(tokenizer, prefix_length, max_len, atk_what='prefix', seed=42):
    # Load the datasets
    jailbreak_data_one = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_05_07', split='train')
    jailbreak_data_two = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_12_25', split='train')
    regular_data_one = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_05_07', split='train')
    regular_data_two = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_12_25', split='train')

    # Concatenate the jailbreak and regular datasets separately
    jailbreak_data = concatenate_datasets([jailbreak_data_one, jailbreak_data_two])
    regular_data = concatenate_datasets([regular_data_one, regular_data_two])

    # Determine the number of samples in each class
    num_jailbreak = len(jailbreak_data)
    num_regular = len(regular_data)

    # Calculate the ratio of each class
    total_samples = num_jailbreak + num_regular
    ratio_jailbreak = num_jailbreak / total_samples
    ratio_regular = num_regular / total_samples

    # Calculate the number of validation and evaluation samples per class
    num_val_jailbreak = int(1000 * ratio_jailbreak)
    num_val_regular = 1000 - num_val_jailbreak
    num_eval_jailbreak = int(1000 * ratio_jailbreak)
    num_eval_regular = 1000 - num_eval_jailbreak

    # Ensure we don't pick more samples than available
    num_val_jailbreak = min(num_val_jailbreak, num_jailbreak)
    num_val_regular = min(num_val_regular, num_regular)
    num_eval_jailbreak = min(num_eval_jailbreak, num_jailbreak - num_val_jailbreak)
    num_eval_regular = min(num_eval_regular, num_regular - num_val_regular)

    # Create indices for each class
    jailbreak_indices = np.arange(num_jailbreak)
    regular_indices = np.arange(num_regular)

    # Split indices for jailbreak data
    train_jailbreak_indices, val_eval_jailbreak_indices = train_test_split(
        jailbreak_indices, test_size=(num_val_jailbreak + num_eval_jailbreak), random_state=42, shuffle=True
    )
    val_jailbreak_indices, eval_jailbreak_indices = train_test_split(
        val_eval_jailbreak_indices, test_size=num_eval_jailbreak / (num_val_jailbreak + num_eval_jailbreak) if (num_val_jailbreak + num_eval_jailbreak) > 0 else 0.5, random_state=42, shuffle=True
    )

    # Split indices for regular data
    train_regular_indices, val_eval_regular_indices = train_test_split(
        regular_indices, test_size=(num_val_regular + num_eval_regular), random_state=42, shuffle=True
    )
    val_regular_indices, eval_regular_indices = train_test_split(
        val_eval_regular_indices, test_size=num_eval_regular / (num_val_regular + num_eval_regular) if (num_val_regular + num_eval_regular) > 0 else 0.5, random_state=42, shuffle=True
    )

    # Select data based on the indices
    train_data = concatenate_datasets([
        jailbreak_data.select(train_jailbreak_indices),
        regular_data.select(train_regular_indices)
    ]).shuffle(seed=42)

    validation_data = concatenate_datasets([
        jailbreak_data.select(val_jailbreak_indices),
        regular_data.select(val_regular_indices)
    ]).shuffle(seed=42)

    evaluation_data = concatenate_datasets([
        jailbreak_data.select(eval_jailbreak_indices),
        regular_data.select(eval_regular_indices)
    ]).shuffle(seed=42)
        
    train_data = train_data.map(
        lambda examples: preprocess_function(examples, prefix_length, max_len, tokenizer, atk_what),
        batched=True
    )
    validation_data = validation_data.map(
        lambda examples: preprocess_function(examples, prefix_length, max_len, tokenizer, atk_what),
        batched=True
    )
    evaluation_data = evaluation_data.map(
        lambda examples: preprocess_function(examples, prefix_length, max_len, tokenizer, atk_what),
        batched=True
    )
    
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    validation_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    evaluation_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Define the batch size
    batch_size = 16  # You can adjust this as needed
    # Create DataLoaders
    # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False) # No need to shuffle validation data
    evaluation_dataloader = DataLoader(evaluation_data, batch_size=1, shuffle=False) # No need to shuffle evaluation data
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(validation_data)}")
    print(f"Evaluation data size: {len(evaluation_data)}")
    
    print("\nClass distribution in each split:")
    print(f"Training: {get_class_distribution(train_data)}")
    print(f"Validation: {get_class_distribution(validation_data)}")
    print(f"Evaluation: {get_class_distribution(evaluation_data)}")
    
    print(f"Number of batches in training dataloader: {len(train_dataloader)}")
    print(f"Number of batches in validation dataloader: {len(validation_dataloader)}")
    print(f"Number of batches in evaluation dataloader: {len(evaluation_dataloader)}")
    
    return train_dataloader, validation_dataloader, evaluation_dataloader

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

# def get_influences(true_class_id, predictions, probs):
#     '''
#     Calculate influence scores for binary classification
#     '''
#     # Fix typo in variable name
#     influences = []
    
#     # Get source probability for true class
#     src_prob = probs[0] if predictions[0] == true_class_id else 1 - probs[0]
    
#     # Calculate influence for each prediction
#     for i, pred in enumerate(predictions):
#         curr_prob = probs[i] if pred == true_class_id else 1 - probs[i]
#         influence = curr_prob - src_prob
#         influences.append(influence)  # Fixed typo in append
        
#     return influences

# def process_single_document(input_id, attention_mask, model, true_class_id, tokenizer, num_masks, mask_token_id):
#     # Convert tensors to lists for hashing
#     input_id_list = input_id.cpu().tolist()
#     attention_mask_list = attention_mask.cpu().tolist()

#     # Create the cache key from the input parameters
#     key = hashlib.sha256(json.dumps({
#         "input_id_list": input_id_list,
#         "attention_mask_list": attention_mask_list,
#         "true_class_id": true_class_id,
#         "num_masks": num_masks,
#         "mask_token_id": mask_token_id,
#     }, sort_keys=True).encode()).hexdigest()

#     # Check if the result is already in the cache
#     if key in cache:
#         cached_result = cache[key]
#         # Ensure the retrieved tensor is on the correct device
#         masked_input = torch.from_numpy(cached_result[0]).to(input_id.device)  # Move to the device of the input
#         return masked_input, cached_result[1]

#     # Find valid positions (non-padding tokens)
#     valid_positions = [i for i, val in enumerate(attention_mask_list) if val == 1]

#     # Skip if not enough valid positions
#     if len(valid_positions) < num_masks:
#         num_masks = len(valid_positions)-1

#     # Create batch inputs for this sample
#     sample_size = len(valid_positions) + 1  # +1 for original
#     sample_input_ids = [input_id_list.copy() for _ in range(sample_size)]
#     for j, pos in enumerate(valid_positions):
#         sample_input_ids[j + 1][pos] = mask_token_id

#     sample_input_tensor = torch.tensor(sample_input_ids)
#     sample_docs = tokenizer.batch_decode(sample_input_tensor, skip_special_tokens=True)

#     # Run batch inference
#     _, predictions_, probs_ = get_raw_logits.process_file(data=sample_docs)

#     # Calculate token influences
#     influences = get_influences(true_class_id, predictions_, probs_)

#     # Get indices of top influential tokens
#     top_indices = sorted(range(1, len(influences)), key=lambda x: influences[x], reverse=True)[:num_masks]

#     # Map back to token positions
#     chosen_positions = [valid_positions[j - 1] for j in top_indices]

#     # Create masked version
#     masked_input = input_id.clone()
#     for pos in chosen_positions:
#         masked_input[pos] = mask_token_id
    
#     # Store the result in the cache, converting the tensor to a NumPy array
#     cache[key] = (masked_input.cpu().numpy(), chosen_positions) # Changed this line
#     return masked_input, chosen_positions


# def apply_importance_masks(input_ids, attention_mask, model, true_class_ids, tokenizer, num_masks=10, mask_token_id=None):
#     batch_size = input_ids.size(0)
#     masked_input_ids = input_ids.clone()
#     mask_positions = []

#     # inner bar on its own line (position=1), will be cleared when done
#     inner = tqdm(
#         range(batch_size),
#         desc="Processing batch influences",
#         position=1,
#         leave=False
#     )
#     for i in inner:
#         masked_input, positions = process_single_document(
#             input_ids[i],
#             attention_mask[i],
#             model,
#             true_class_ids[i],
#             tokenizer,
#             num_masks,
#             mask_token_id
#         )
#         mask_positions.append(positions)
#     return masked_input_ids, mask_positions

def apply_importance_masks(input_ids, attention_mask, model, true_class_ids, tokenizer, num_masks=10, mask_token_id=None):
    batch_size = input_ids.size(0)
    masked_input_ids = input_ids.clone()
    mask_positions = []

    for i in range(batch_size):
        masked_input, positions = process_single_document(
            input_ids[i],
            attention_mask[i],
            model,
            true_class_ids[i],
            tokenizer,
            num_masks,
            mask_token_id
        )
    
        mask_positions.append(positions)
    # cache_report()
    return masked_input_ids, mask_positions


def logits_to_labels_prefix(input_ids, prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, prefix_length=10):
    """Original function for prefix-based approach"""
    batch_size = prefix_logits.size(0)
    num_prefix_tokens = prefix_logits.size(1)
    vocab_size = prefix_logits.size(2)
    
    # Sample a token for each masked prefix position based on the probabilities
    prefix_probabilities = F.softmax(prefix_logits, dim=-1) # [batch_size, prefix_length, vocab_size]
    sampled_prefix_tokens = torch.multinomial(prefix_probabilities.view(-1, vocab_size), num_samples=1).view(batch_size, num_prefix_tokens) # [batch_size, prefix_length]
    prefix_labels = sampled_prefix_tokens.clone().detach()
    
    # Convert sampled prefix token IDs and original input IDs back to text for reward calculation
    generated_prefix_tokens_list = sampled_prefix_tokens.tolist()
    original_input_ids_list = input_ids.tolist()
    source_documents = []
    generated_documents = []

    for i in range(batch_size):            
        # Get the original document (excluding the prefix of [MASK] tokens)
        original_document_tokens = [token_id for idx, token_id in enumerate(original_input_ids_list[i]) if idx >= prefix_length and token_id != pad_token_id and token_id not in [cls_token_id, sep_token_id]]
        original_document = tokenizer.decode(original_document_tokens, skip_special_tokens=True)

        # Combine generated prefix and original document
        generated_prefix = tokenizer.decode(generated_prefix_tokens_list[i], skip_special_tokens=True)
        generated_document = generated_prefix + " " + original_document # You might want a different way to combine
        source_documents.append(original_document)
        generated_documents.append(generated_document)
        
    return batch_size, num_prefix_tokens, vocab_size, prefix_labels, source_documents, generated_documents

def logits_to_labels_doc(input_ids, masked_positions, token_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id):
    """New function for document-based approach"""
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
          
def main(args):
    atker_path = args.atker_path
    target_path = args.target_path
    len_doc_max = args.len_doc_max
    prefix_length = args.prefix_length
    save_to_path = args.save_to_path
    atk_what = args.atk_what

    # atk_pattern = 'influence'
    atk_pattern = 'random'
    
    # Validate attack strategy
    if atk_what not in ['prefix', 'doc']:
        raise ValueError(f"Invalid attack strategy: {atk_what}. Must be 'prefix' or 'doc'.")
    
    # Number of random tokens to mask in document mode
    num_doc_masks = args.num_doc_masks if hasattr(args, 'num_doc_masks') else prefix_length

    best_gpu = GPUtil.getFirstAvailable(order='memoryFree', maxLoad=0.8, maxMemory=0.8)[0]
    torch.cuda.set_device(best_gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    
    print(f"Using GPU {best_gpu}")

    # if torch.cuda.is_available():
    #     device = torch.device(f'cuda:{0}')  # Just use the first available GPU
    #     torch.cuda.set_device(device)
    # else:
    #     device = torch.device('cpu')
    # print(f"Using GPU {device}")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
     

    USE = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")

    tokenizer = AutoTokenizer.from_pretrained(atker_path, max_length=len_doc_max)
    vocab_size = tokenizer.vocab_size
    print(f'Attacker Base: {atker_path}')
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Attack strategy: {atk_what}")

    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id

    train_dataloader, validation_dataloader, evaluation_dataloader = build_datasets(
        tokenizer=tokenizer, 
        prefix_length=prefix_length, 
        max_len=len_doc_max, 
        atk_what=atk_what,
        seed=42
    )
    
    # Initialize model
    config = BertConfig.from_pretrained(atker_path, output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained(atker_path, config=config).to(device)
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
    
    # Training function
    def train(model, train_dataloader, validation_dataloader, evaluation_dataloader, tokenizer, optimizer, epochs, eval_interval=100, print_length=200):  
        model.train()
        best_acc, val_accuracy = float('inf'), float('inf')
        step = -1

        total_batches = len(train_dataloader)
        total_steps = epochs * total_batches

        # single bar over all steps
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
                
                # Process differently based on attack strategy
                if atk_what == 'prefix':
                    # Original prefix-based approach
                    logits = model(input_ids, attention_mask).logits
                    prefix_logits = logits[:, :prefix_length, :]
                    batch_size, num_prefix_tokens, vocab_size, prefix_labels, source_documents, generated_documents = logits_to_labels_prefix(
                        input_ids, prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, prefix_length=prefix_length
                    )
                    
                    # Calculate rewards using Llama Guard
                    rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
                    prompt_, predictions_, probs_ = get_raw_logits.process_file(data=generated_documents)
                    labels_all.extend(predictions_)
                    predicted_classes = torch.tensor(predictions_).to(device)
                    
                    for i, gen_doc in enumerate(generated_documents):
                        if predictions_[i] not in (1, 0):
                            if labels[i] == 0:
                                rewards[i] = 0.0
                            else:
                                rewards[i] = 1.0
                        elif labels[i] == predictions_[i]:
                            rewards[i] = 1 - probs_[i]
                        else:
                            rewards[i] = probs_[i]
                    
                    # Calculate loss for prefix-based approach
                    loss = torch.tensor(0.0, device=device)
                    for i in range(batch_size):
                        batch_custom_loss = F.cross_entropy(
                            prefix_logits[i],  # Logits for the prefix of the i-th document [prefix_length, vocab_size]
                            prefix_labels[i],  # Sampled tokens for the prefix of the i-th document [prefix_length]
                        )
                        loss += batch_custom_loss * rewards[i]
                    
                elif atk_what == 'doc':
                    if atk_pattern == 'random':
                        masked_input_ids, mask_positions = apply_random_masks(
                            input_ids, attention_mask, num_masks=num_doc_masks, mask_token_id=mask_token_id
                        )
                    elif atk_pattern == 'influence':
                        true_class_ids = labels.tolist()
                        masked_input_ids, mask_positions = apply_importance_masks(
                            input_ids, attention_mask, model, true_class_ids, tokenizer=tokenizer,
                            num_masks=num_doc_masks, mask_token_id=mask_token_id)

                    # 2. Get logits from the model for these masked positions
                    logits = model(masked_input_ids, attention_mask).logits
                    
                    # 3. Sample new tokens for the masked positions and calculate loss
                    batch_size, vocab_size, generated_labels, source_documents, generated_documents = logits_to_labels_doc(
                        input_ids, mask_positions, logits, tokenizer, pad_token_id, cls_token_id, sep_token_id
                    )
                    
                    # 4. Calculate rewards using Llama Guard
                    adv_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
                    sem_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)

                    prompt_, predictions_, probs_ = get_raw_logits.process_file(data=generated_documents)
                    labels_all.extend(predictions_)
                    predicted_classes = torch.tensor(predictions_).to(device)
                    
                    for i, gen_doc in enumerate(generated_documents):                        
                        if predictions_[i] not in (1, 0):
                            if labels[i] == 0:
                                adv_rewards[i]  0.0
                            else:
                                adv_rewards[i] = 1.0
                        elif labels[i] == predictions_[i]:
                            adv_rewards[i] = 1 - probs_[i]
                        else:
                            adv_rewards[i] = probs_[i]
                        sem_rewards[i] = getUSEcosSimilarity(prompt_[i], gen_doc)
                        rewards = adv_rewards + sem_rewards
                    
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
                
                # Store documents for logging
                source_documents_all.extend(source_documents)
                generated_documents_all.extend(generated_documents)
                
                # Normalize loss by batch size
                loss /= batch_size
                
                # Backpropagate
                loss.backward()
                optimizer.step()

                # Update progress bar
                bar.update(1)
                bar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "batch": f"{batch_idx}/{total_batches}",
                    "loss": f"{loss.item():.4f}"
                })
                
                # Evaluation
                if step % eval_interval == 0:
                    print('evaluating...')
                    model.eval()
                    val_loss_total = 0
                    val_batches = len(validation_dataloader)

                    acc_metric = Accuracy(task="binary").to(device)
                    val_acc_metric = Accuracy(task="binary").to(device)
                    
                    with torch.no_grad():
                        for val_batch in validation_dataloader:
                            val_input_ids = val_batch['input_ids'].to(device)
                            val_attention_mask = val_batch['attention_mask'].to(device)
                            val_labels = val_batch['labels'].to(device)
                            
                            if atk_what == 'prefix':
                                val_logits = model(val_input_ids, val_attention_mask).logits
                                val_prefix_logits = val_logits[:, :prefix_length, :]
                                _, _, _, _, val_source_documents, val_generated_documents = logits_to_labels_prefix(
                                    val_input_ids, val_prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, prefix_length=prefix_length
                                )
                            elif atk_what == 'doc':
                                if atk_pattern == 'random':
                                    val_masked_input_ids, val_mask_positions = apply_random_masks(
                                        val_input_ids, val_attention_mask, num_masks=num_doc_masks, mask_token_id=mask_token_id
                                    )
                                elif atk_pattern == 'influence':
                                    true_class_ids = val_labels.tolist()
                                    val_masked_input_ids, val_mask_positions = apply_importance_masks(
                                        val_input_ids, val_attention_mask, model, true_class_ids, tokenizer=tokenizer,
                                        num_masks=num_doc_masks, mask_token_id=mask_token_id
                                    )
                                val_logits = model(val_masked_input_ids, val_attention_mask).logits
                                _, _, _, val_source_documents, val_generated_documents = logits_to_labels_doc(
                                    val_input_ids, val_mask_positions, val_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id
                                )
                                
                            val_source_documents_all.extend(val_source_documents) 
                            val_generated_documents_all.extend(val_generated_documents)
                            val_prompt_, val_predictions_, val_probs_ = get_raw_logits.process_file(data=val_generated_documents)
                            val_labels_all.extend(val_labels.tolist())

                            val_predicted_classes = torch.tensor(val_predictions_).to(device)
                            acc_metric.update(predicted_classes, labels)
                            val_acc_metric.update(val_predicted_classes, val_labels)

                        accuracy = acc_metric.compute()
                        val_accuracy = val_acc_metric.compute()

                        print(
                            f"Step: {step}, Training Loss: {loss.item():.4f}, "
                            f"Training Accuracy: {accuracy:.4f}, "
                            f"Validation Accuracy: {val_accuracy:.4f}, "
                        )
                        
                        # Print examples of generated documents
                        if source_documents_all:
                            train_idx = random.randint(0, len(source_documents_all) - 1)
                            print('train src: ', source_documents_all[train_idx][:print_length])
                            print('train gen: ', generated_documents_all[train_idx][:print_length])
                            print('train lab: ', labels_all[train_idx])
                        
                        if val_source_documents_all:
                            val_idx = random.randint(0, len(val_source_documents_all) - 1)
                            print('val src: ', val_source_documents_all[val_idx][:print_length])
                            print('val gen: ', val_generated_documents_all[val_idx][:print_length])
                            print('val lab: ', val_labels_all[min(train_idx, len(val_labels_all)-1)])
                        
                        # Clear stored documents to free memory
                        source_documents_all.clear()
                        generated_documents_all.clear()
                        val_source_documents_all.clear()
                        val_generated_documents_all.clear()
                        
                        # Reset metrics for the next validation stage
                        acc_metric.reset()
                        val_acc_metric.reset()
                    
                    model.train()
                    
                    # Save model if it improves
                    if val_accuracy < best_acc:
                        best_acc = val_accuracy
                        model_path = f"{save_to_path}/attacker_{atk_what}_{timestamp}_llama-guard_{epoch}_{step}_{best_acc:.4f}.pth"
                        torch.save(model.state_dict(), model_path)
                        print(f"Model saved at step {step} with accuracy: {best_acc:.4f}, path: {model_path}")
        
        bar.close()

    # Start training
    train(model, train_dataloader, validation_dataloader, evaluation_dataloader, tokenizer, optimizer, epochs=10, eval_interval=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to generate adversarial tokens")
    parser.add_argument('--atker_path', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--target_path', type=str, required=True, help='target model path')
    parser.add_argument('--save_to_path', type=str, required=True, help='target model path')
    parser.add_argument('--len_doc_max', type=int, default=512, help='max length of document')
    parser.add_argument('--prefix_length', type=int, default=10, help='length of prefix for prefix-based attack')
    parser.add_argument('--atk_what', type=str, default='prefix', choices=['prefix', 'doc'], 
                        help='attack strategy: "prefix" (original) or "doc" (modify document tokens)')
    parser.add_argument('--num_doc_masks', type=int, default=10, 
                        help='number of tokens to mask in document when using "doc" strategy')

    # args = parser.parse_args()    

    args = argparse.Namespace(
        atker_path='bert-base-uncased', # Example path
        target_path='temp',
        len_doc_max=512,
        prefix_length=10,
        save_to_path='/usa/taikun/07_transencoder/1training/llama-guard-attacker',
        atk_what='doc',
        num_doc_masks=10)
    
    
    print_config(args)
    main(args)