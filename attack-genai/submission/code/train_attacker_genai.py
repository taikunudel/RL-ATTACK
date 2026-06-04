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
import hashlib
import diskcache
cache = diskcache.Cache('./cache', size_limit=10e9)
cache.stats(enable=True)    
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import train_test_split
# import GPUtil  # Not actually used in the code
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
print(f"Using GPU {device}")

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
    
def preprocess_function(examples, num_doc_masks, len_doc_max, tokenizer, atk_what='prefix'):
    """
    Preprocess function that handles different attack types:
    - 'prefix': Add [MASK] tokens as a prefix (original functionality)
    - 'doc': Keep original text but insert placeholders for later masking within document
    """
    if atk_what == 'prefix':
        # Original prefix-based approach
        mask_token = '[MASK]'
        prefix = ' '.join([mask_token] * num_doc_masks)
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

def build_datasets(tokenizer, num_doc_masks, max_len, atk_what='prefix', seed=42):
    # Load the datasets
    jailbreak_data_one = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_05_07', split='train')
    jailbreak_data_two = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_12_25', split='train')
    regular_data_one = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_05_07', split='train')
    regular_data_two = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_12_25', split='train')

    # Concatenate the jailbreak and regular datasets separately
    jailbreak_data = concatenate_datasets([jailbreak_data_one, jailbreak_data_two])
    regular_data = concatenate_datasets([regular_data_one, regular_data_two])

    # Filter out data longer than max_len tokens BEFORE splitting
    def filter_by_length(example):
        return len(tokenizer.tokenize(example['prompt'])) <= max_len

    jailbreak_data = jailbreak_data.filter(filter_by_length)
    regular_data = regular_data.filter(filter_by_length)

    # Determine the number of samples in each class after filtering
    num_jailbreak = len(jailbreak_data)
    num_regular = len(regular_data)
    total_samples = num_jailbreak + num_regular

    # Calculate the class ratio in the full dataset (which will be our target ratio)
    ratio_jailbreak = num_jailbreak / total_samples if total_samples > 0 else 0.5
    ratio_regular = num_regular / total_samples if total_samples > 0 else 0.5

    # Calculate the number of validation samples per class to maintain the ratio
    num_val_jailbreak_target = int(100 * ratio_jailbreak)
    num_val_regular_target = 100 - num_val_jailbreak_target

    # Ensure we don't pick more validation samples than available
    num_val_jailbreak = min(num_val_jailbreak_target, num_jailbreak)
    num_val_regular = min(num_val_regular_target, num_regular)

    # Calculate the number of evaluation samples per class to maintain the ratio (up to 1000)
    num_eval_jailbreak_target = int(1000 * ratio_jailbreak)
    num_eval_regular_target = 1000 - num_eval_jailbreak_target

    # Ensure we don't pick more evaluation samples than available after validation split
    remaining_jailbreak = num_jailbreak - num_val_jailbreak
    remaining_regular = num_regular - num_val_regular
    num_eval_jailbreak = min(num_eval_jailbreak_target, remaining_jailbreak)
    num_eval_regular = min(num_eval_regular_target, remaining_regular)
    
    # Create indices for each class
    jailbreak_indices = np.arange(num_jailbreak)
    regular_indices = np.arange(num_regular)

    # Split indices for jailbreak data
    train_jailbreak_indices, val_eval_jailbreak_indices = train_test_split(
        jailbreak_indices, test_size=(num_val_jailbreak + num_eval_jailbreak), random_state=seed, shuffle=True
    )
    val_jailbreak_indices, eval_jailbreak_indices = train_test_split(
        val_eval_jailbreak_indices, test_size=num_eval_jailbreak / (num_val_jailbreak + num_eval_jailbreak) if (num_val_jailbreak + num_eval_jailbreak) > 0 else 0.5, random_state=seed, shuffle=True
    )

    # Split indices for regular data
    train_regular_indices, val_eval_regular_indices = train_test_split(
        regular_indices, test_size=(num_val_regular + num_eval_regular), random_state=seed, shuffle=True
    )
    val_regular_indices, eval_regular_indices = train_test_split(
        val_eval_regular_indices, test_size=num_eval_regular / (num_val_regular + num_eval_regular) if (num_val_regular + num_eval_regular) > 0 else 0.5, random_state=seed, shuffle=True
    )

    # Select data based on the indices
    train_data = concatenate_datasets([
        jailbreak_data.select(train_jailbreak_indices),
        regular_data.select(train_regular_indices)
    ]).shuffle(seed=seed)

    validation_data = concatenate_datasets([
        jailbreak_data.select(val_jailbreak_indices),
        regular_data.select(val_regular_indices)
    ]).shuffle(seed=seed)

    evaluation_data = concatenate_datasets([
        jailbreak_data.select(eval_jailbreak_indices),
        regular_data.select(eval_regular_indices)
    ]).shuffle(seed=seed)

    train_data = train_data.map(
        lambda examples: preprocess_function(examples, num_doc_masks, max_len, tokenizer, atk_what),
        batched=True
    )
    validation_data = validation_data.map(
        lambda examples: preprocess_function(examples, num_doc_masks, max_len, tokenizer, atk_what),
        batched=True
    )
    evaluation_data = evaluation_data.map(
        lambda examples: preprocess_function(examples, num_doc_masks, max_len, tokenizer, atk_what),
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

def logits_to_labels_prefix(input_ids, prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, num_doc_masks=10):
    """Original function for prefix-based approach"""
    batch_size = prefix_logits.size(0)
    num_prefix_tokens = prefix_logits.size(1)
    vocab_size = prefix_logits.size(2)
    
    # Sample a token for each masked prefix position based on the probabilities
    prefix_probabilities = F.softmax(prefix_logits, dim=-1) # [batch_size, num_doc_masks, vocab_size]
    sampled_prefix_tokens = torch.multinomial(prefix_probabilities.view(-1, vocab_size), num_samples=1).view(batch_size, num_prefix_tokens) # [batch_size, num_doc_masks]
    prefix_labels = sampled_prefix_tokens.clone().detach()
    
    # Convert sampled prefix token IDs and original input IDs back to text for reward calculation
    generated_prefix_tokens_list = sampled_prefix_tokens.tolist()
    original_input_ids_list = input_ids.tolist()
    source_documents = []
    generated_documents = []

    for i in range(batch_size):            
        # Get the original document (excluding the prefix of [MASK] tokens)
        original_document_tokens = [token_id for idx, token_id in enumerate(original_input_ids_list[i]) if idx >= num_doc_masks and token_id != pad_token_id and token_id not in [cls_token_id, sep_token_id]]
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

class MaskFillingHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(MaskFillingHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, vocab_size)  # Output to vocab size for mask filling

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class BertForJailbreak(nn.Module):
    def __init__(self, atker_path):
        super(BertForJailbreak, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(atker_path, output_hidden_states=True)
        self.config = BertConfig.from_pretrained(atker_path)
        embedding_dim = self.config.hidden_size
        hidden_dim = 256
        self.classification_head = MaskFillingHead(embedding_dim, hidden_dim, self.bert.config.vocab_size).to(device)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        logits = self.classification_head(outputs.last_hidden_state)
        return logits

          
def main(args):
    atker_path = args.atker_path
    target_path = args.target_path
    len_doc_max = args.len_doc_max
    num_doc_masks = args.num_doc_masks
    save_to_path = args.save_to_path
    atk_what = args.atk_what
    linear_head = args.linear_head
    alpha = args.alpha
    server_url = args.server_url

    # atk_pattern = 'influence'
    atk_pattern = 'random'
    
    # Validate attack strategy
    if atk_what not in ['prefix', 'doc']:
        raise ValueError(f"Invalid attack strategy: {atk_what}. Must be 'prefix' or 'doc'.")
    
    # Number of random tokens to mask in document mode
    num_doc_masks = args.num_doc_masks if hasattr(args, 'num_doc_masks') else num_doc_masks

    # best_gpu = GPUtil.getFirstAvailable(order='memoryFree', maxLoad=0.8, maxMemory=0.8)[0]
    # torch.cuda.set_device(best_gpu)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    
    # print(f"Using GPU {best_gpu}")

    

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
        num_doc_masks=num_doc_masks, 
        max_len=len_doc_max, 
        atk_what=atk_what,
        seed=42
    )

    if linear_head:
        print('Using MLM Head (BertForMaskedLM built-in)')
        config = BertConfig.from_pretrained(atker_path, output_hidden_states=True)
        model = BertForMaskedLM.from_pretrained(atker_path, config=config).to(device)
        # Freeze all layers except the MLM head
        for name, param in model.named_parameters():
            if 'cls' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        print('Using MaskFillingHead (3-layer FFN)')
        model = BertForJailbreak(atker_path).to(device) # use new class
        for name, param in model.named_parameters():
            if 'classification_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Resume from checkpoint if specified
    resume_epoch, resume_step, resume_best_acc = 0, -1, float('inf')
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            resume_epoch = ckpt['epoch']
            resume_step = ckpt['step']
            resume_best_acc = ckpt['best_acc']
            print(f"Resumed full checkpoint: epoch={resume_epoch}, step={resume_step}, best_acc={resume_best_acc:.4f}")
        else:
            model.load_state_dict(ckpt)
            print(f"Resumed model weights from: {args.resume_from}")

    # Training function
    def train(model, train_dataloader, validation_dataloader, evaluation_dataloader, tokenizer, optimizer, epochs, eval_interval=100, print_length=200, start_epoch=0, start_step=-1, best_acc_init=float('inf')):
        model.train()
        best_acc, val_accuracy = best_acc_init, float('inf')
        step = start_step

        total_batches = len(train_dataloader)
        total_steps = epochs * total_batches

        bar = tqdm(total=total_steps, initial=step+1, desc="Training", unit="step")
        source_documents_all, generated_documents_all, labels_all = [], [], []
        val_source_documents_all, val_generated_documents_all, val_labels_all = [], [], []
        reward_history, adv_reward_history, sem_reward_history = [], [], []

        for epoch in range(start_epoch, epochs):
            for batch_idx, batch in enumerate(train_dataloader, start=1):
                step += 1
                # Skip steps already completed in a previous run
                if step <= start_step:
                    continue
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                if atk_what == 'prefix':
                    logits = model(input_ids, attention_mask).logits
                    prefix_logits = logits[:, :num_doc_masks, :]
                    batch_size, num_prefix_tokens, vocab_size, prefix_labels, source_documents, generated_documents = logits_to_labels_prefix(
                        input_ids, prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, num_doc_masks=num_doc_masks
                    )
                    
                    rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
                    _, predictions_, probs_ = get_raw_logits.process_file(data=generated_documents, server_url=server_url)
                    labels_all.extend(predictions_)
                    predicted_classes = torch.tensor(predictions_).to(device)
                    
                    # for i, gen_doc in enumerate(generated_documents):
                    #     if predictions_[i] not in (1, 0):
                    #         if labels[i] == 0:
                    #             rewards[i] = 0.0
                    #         else:
                    #             rewards[i] = 1.0
                    #     elif labels[i] == predictions_[i]:
                    #         rewards[i] = 1 - probs_[i]
                    #     else:
                    #         rewards[i] = probs_[i]
                    
                    # # Calculate loss for prefix-based approach
                    # loss = torch.tensor(0.0, device=device)
                    # for i in range(batch_size):
                    #     batch_custom_loss = F.cross_entropy(
                    #         prefix_logits[i],  # Logits for the prefix of the i-th document [num_doc_masks, vocab_size]
                    #         prefix_labels[i],  # Sampled tokens for the prefix of the i-th document [num_doc_masks]
                    #     )
                    #     loss += batch_custom_loss * rewards[i]
                    
                elif atk_what == 'doc':
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

                        sem_rewards[i] = USE_res[i]
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
                postfix = {
                    "epoch": f"{epoch+1}/{epochs}",
                    "batch": f"{batch_idx}/{total_batches}",
                    "loss": f"{loss.item():.4f}",
                }
                if atk_what == 'doc':
                    postfix["adv_r"] = f"{adv_rewards.mean().item():.3f}"
                    postfix["sem_r"] = f"{sem_rewards.mean().item():.3f}"
                bar.set_postfix(postfix)

                # Track rewards
                if atk_what == 'doc':
                    reward_history.append(rewards.mean().item())
                    adv_reward_history.append(adv_rewards.mean().item())
                    sem_reward_history.append(sem_rewards.mean().item())

                # Evaluation
                if step % eval_interval == 0:
                    print('evaluating...')
                    model.eval()
                    val_loss_total = 0
                    val_batches = len(validation_dataloader)

                    val_acc_metric = Accuracy(task="binary").to(device)
                    
                    with torch.no_grad():
                        for val_batch in validation_dataloader:
                            val_input_ids = val_batch['input_ids'].to(device)
                            val_attention_mask = val_batch['attention_mask'].to(device)
                            val_labels = val_batch['labels'].to(device)
                            
                            if atk_what == 'prefix':
                                val_logits = model(val_input_ids, val_attention_mask).logits
                                val_prefix_logits = val_logits[:, :num_doc_masks, :]
                                _, _, _, _, val_source_documents, val_generated_documents = logits_to_labels_prefix(
                                    val_input_ids, val_prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, num_doc_masks=num_doc_masks
                                )
                            elif atk_what == 'doc':
                                val_masked_input_ids, val_mask_positions = apply_random_masks(
                                    val_input_ids, val_attention_mask, num_masks=num_doc_masks, mask_token_id=mask_token_id
                                )
                                val_logits = model(val_masked_input_ids, val_attention_mask).logits
                                _, _, _, val_source_documents, val_generated_documents = logits_to_labels_doc(
                                    val_input_ids, val_mask_positions, val_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id
                                )
                                
                            val_source_documents_all.extend(val_source_documents) 
                            val_generated_documents_all.extend(val_generated_documents)
                            val_prompt_, val_predictions_, val_probs_ = get_raw_logits.process_file(data=val_generated_documents, server_url=server_url)
                            val_labels_all.extend(val_labels.tolist())

                            val_predicted_classes = torch.tensor(val_predictions_).to(device)
                            val_acc_metric.update(val_predicted_classes, val_labels)

                        val_accuracy = val_acc_metric.compute()

                        print(
                            f"Step: {step}, Training Loss: {loss.item():.4f}, "
                            f"Validation Accuracy: {val_accuracy:.4f}, "
                        )

                        # Print reward statistics
                        if reward_history:
                            n = min(eval_interval, len(reward_history))
                            print(
                                f"  Mean rewards (last {n} steps): "
                                f"total={np.mean(reward_history[-n:]):.4f}, "
                                f"adv={np.mean(adv_reward_history[-n:]):.4f}, "
                                f"sem={np.mean(sem_reward_history[-n:]):.4f}"
                            )

                        # Print examples of generated documents
                        if source_documents_all:
                            train_idx = random.randint(0, len(source_documents_all) - 1)
                            print('train src: ', source_documents_all[train_idx][:print_length])
                            print('train gen: ', generated_documents_all[train_idx][:print_length])
                            if train_idx < len(labels_all):
                                print('train lab: ', labels_all[train_idx])

                        if val_source_documents_all:
                            val_idx = random.randint(0, len(val_source_documents_all) - 1)
                            print('val src: ', val_source_documents_all[val_idx][:print_length])
                            print('val gen: ', val_generated_documents_all[val_idx][:print_length])
                            print('val lab: ', val_labels_all[min(val_idx, len(val_labels_all)-1)])

                        # Clear stored documents to free memory
                        source_documents_all.clear()
                        generated_documents_all.clear()
                        labels_all.clear()
                        val_source_documents_all.clear()
                        val_generated_documents_all.clear()
                        val_labels_all.clear()

                        # Reset metrics for the next validation stage
                        val_acc_metric.reset()
                    
                    model.train()
                    
                    # Save best model if it improves
                    if val_accuracy < best_acc:
                        best_acc = val_accuracy
                        model_path = f"{save_to_path}/attacker_{timestamp}_llama-guard_{atk_what}_{alpha}_{epoch}_{step}_{best_acc:.4f}.pth"
                        torch.save(model.state_dict(), model_path)
                        print(f"Best model saved at step {step} with accuracy: {best_acc:.4f}, path: {model_path}")

                    # Always save last checkpoint for resuming
                    last_path = f"{save_to_path}/attacker_{timestamp}_llama-guard_{atk_what}_{alpha}_last.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'best_acc': best_acc,
                    }, last_path)
        
        bar.close()

    # Start training
    train(model, train_dataloader, validation_dataloader, evaluation_dataloader, tokenizer, optimizer, epochs=10, eval_interval=100, start_epoch=resume_epoch, start_step=resume_step, best_acc_init=resume_best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to generate adversarial tokens")
    parser.add_argument('--atker_path', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--target_path', type=str, required=True, help='target model path')
    parser.add_argument('--save_to_path', type=str, required=True, help='target model path')
    parser.add_argument('--len_doc_max', type=int, default=512, help='max length of document')
    parser.add_argument('--atk_what', type=str, default='prefix', choices=['prefix', 'doc'], help='attack strategy: "prefix" (original) or "doc" (modify document tokens)')
    parser.add_argument('--num_doc_masks', type=int, default=10,  help='number of tokens to mask in document')
    parser.add_argument('--alpha', type=float, default=0.5, help='the weight on the adversarial obj.')
    parser.add_argument('--linear_head', type=bool, default=True, help='Use a linear head instead of a 3-layer FFN')
    parser.add_argument('--server_url', type=str, default="http://localhost:8000/v1", help='8000 for llama guard 3 1B, 8001 for 8B')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')

    args = parser.parse_args()    

    # args = argparse.Namespace(
    #     atker_path='bert-base-uncased', # Example path
    #     target_path='temp',
    #     len_doc_max=512,
    #     num_doc_masks=10,
    #     save_to_path='<redacted-path>
    #     atk_what='doc',
    #     num_doc_masks=10,
    #     alpha=0.5)
    
    
    print_config(args)
    main(args)