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
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
# import GPUtil  # Not actually used in the code
from tqdm.auto import tqdm
from torchmetrics import Accuracy
from transformers import AutoTokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, Dataset, DataLoader
torch.set_default_dtype(torch.float32)
# torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
print(f"Using GPU {device}")

import tensorflow as tf
import tensorflow_hub as hub
# USE (semantic reward) runs on GPU unless USE_CPU=1. GPU keeps the per-step USE
# cost ~ms instead of ~5s on CPU. Cap TF GPU memory growth so it coexists with the
# attacker on the same GPU without grabbing all VRAM.
if os.environ.get("USE_CPU", "0") == "1":
    tf.config.set_visible_devices([], 'GPU')
else:
    try:
        for _g in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass
# # Your TensorFlow code here
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices)) # This should ideally print 1
# if len(physical_devices) > 0:
#     print("Using GPU:", physical_devices[0])

# from utils import *
from rlatk.genai import get_raw_logits
from rlatk.core.similarity_scorer import build_scorer, getUSEcosSimilarity
from rlatk.core.encoders import build_attacker

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def print_config(args):
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

# getUSEcosSimilarity now lives in similarity_scorer.py (imported above).
    
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


def compute_adv_reward(reward_type, label, pred, prob, src_pred=None, src_prob=None):
    """Per-sample adversarial reward r_adv.

    Args:
      label : true class for x (0=safe, 1=unsafe).
      pred  : guard's argmax class on the adversarial x' (0/1).
      prob  : guard's probability on its argmax token for x' (= exp(logprob)).
      src_pred / src_prob : same two quantities on the ORIGINAL x (needed for B/C).

    Variants (plan §3):
      A (default) : the existing validated baseline — UNCHANGED. Reward rises as the
                    guard loses confidence in the true class on x'.
      B           : sigmoid-squashed  σ(Δ),  Δ = P_true(x) − P_true(x')  (bounded, monotone).
      C           : confidence-weighted  w(x)·Δ,  w(x) = guard confidence on the original
                    true class (harder/more-confident originals weighted higher).
    Returns a python float. A is numerically identical to the original inline code.
    """
    lbl = int(label)
    # --- variant A base (identical branches to the original inline reward) ---
    if pred not in (0, 1):
        base = 0.0 if lbl == 0 else 1.0
    elif lbl == pred:
        base = 1.0 - prob
    else:
        base = prob
    if reward_type == 'A' or pred not in (0, 1):
        return float(base)

    # P_true on the adversarial x' (reconstruct true-class prob from argmax prob)
    p_true_adv = prob if pred == lbl else (1.0 - prob)
    # P_true on the original x; if the guard returned a weird token, assume confident-correct
    if src_pred is None or src_pred not in (0, 1):
        p_true_src = 1.0
    else:
        p_true_src = src_prob if src_pred == lbl else (1.0 - src_prob)
    delta = p_true_src - p_true_adv  # drop in true-class prob ∈ [-1, 1]

    if reward_type == 'B':
        return float(1.0 / (1.0 + np.exp(-delta)))        # σ(Δ)
    if reward_type == 'C':
        return float(p_true_src * delta)                  # w(x)·Δ
    return float(base)

# MaskFillingHead and the FFN-head attacker now live in encoders.py (build_attacker).

          
def main(args):
    atker_path = args.atker_path
    target_path = args.target_path
    len_doc_max = args.len_doc_max
    num_doc_masks = args.num_doc_masks
    save_to_path = args.save_to_path
    atk_what = args.atk_what
    linear_head = args.linear_head
    alpha = args.alpha
    reward_type = args.reward_type
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
     

    scorer = build_scorer(args.sim_scorer)

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

    model = build_attacker(atker_path, linear_head=linear_head, device=device)

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
        # Model selection by REWARD (the training objective), since the loss collapses
        # to ~0 in REINFORCE and is useless for picking the best checkpoint. We track a
        # moving average of the combined reward r = alpha*adv + (1-alpha)*sem and save
        # whenever it improves. (Per-step reward is noisy -> use a moving window.)
        best_reward = -float('inf')
        reward_ma = None
        REWARD_MA_BETA = 0.98          # EMA over ~50 steps
        ckpt_path = f"{save_to_path}/attacker_{timestamp}_llama-guard_{atk_what}_{alpha}_{reward_type}_best.pth"
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
                    raise NotImplementedError(
                        "atk_what='prefix' is not implemented (its reward/loss "
                        "were never completed). Use --atk_what doc."
                    )
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
                    
                    USE_res = scorer.score(source_documents, generated_documents)
                    # B/C need the guard's score on the ORIGINAL source docs (one extra
                    # query per sample; free + self-hosted). A keeps the original 1 query.
                    if reward_type in ('B', 'C'):
                        _, src_preds_, src_probs_ = get_raw_logits.process_file(
                            data=source_documents, server_url=server_url)
                    else:
                        src_preds_, src_probs_ = [None] * batch_size, [None] * batch_size
                    for i, gen_doc in enumerate(generated_documents):
                        adv_rewards[i] = compute_adv_reward(
                            reward_type, labels[i].item(), predictions_[i], probs_[i],
                            src_preds_[i], src_probs_[i]
                        )
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

                # Update progress bar — set stats WITHOUT refresh, then a SINGLE
                # update(1) does one redraw per step (avoids the double log line:
                # previously update() + set_postfix() each triggered a redraw).
                postfix = {
                    "epoch": f"{epoch+1}/{epochs}",
                    "batch": f"{batch_idx}/{total_batches}",
                    "loss": f"{loss.item():.4f}",
                }
                if atk_what == 'doc':
                    postfix["adv_r"] = f"{adv_rewards.mean().item():.3f}"
                    postfix["sem_r"] = f"{sem_rewards.mean().item():.3f}"
                bar.set_postfix(postfix, refresh=False)
                bar.update(1)

                # Track rewards
                if atk_what == 'doc':
                    reward_history.append(rewards.mean().item())
                    adv_reward_history.append(adv_rewards.mean().item())
                    sem_reward_history.append(sem_rewards.mean().item())

                    # --- Model selection by REWARD (loss is ~0 in REINFORCE) ---
                    # EMA of the combined objective r = alpha*adv + (1-alpha)*sem.
                    cur_r = rewards.mean().item()
                    reward_ma = cur_r if reward_ma is None else REWARD_MA_BETA * reward_ma + (1 - REWARD_MA_BETA) * cur_r
                    # require a small warmup so the EMA is meaningful, then save best
                    if step >= 20 and reward_ma > best_reward:
                        best_reward = reward_ma
                        torch.save(model.state_dict(), ckpt_path)
                        if step % 50 == 0:
                            print(f"Best model saved at step {step} with accuracy: {best_reward:.4f}, path: {ckpt_path}")

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
                    
                    # Save ONLY the best model, to a STABLE per-run filename so it
                    # overwrites itself (disk-safe: exactly one .pth per run).
                    # (Model selection is now by REWARD inside the step loop, not by
                    # val_accuracy here — the old val-acc save was removed because with a
                    # large eval_interval it only fired at step 0, persisting an UNTRAINED
                    # model. Validation above is still useful for logging val accuracy.)
            # END OF EPOCH fallback: if the reward-based selection never saved
            # (e.g. run shorter than warmup), persist the final model so there's
            # always a valid checkpoint. Does NOT overwrite a reward-best ckpt.
            import os as _os
            if not _os.path.exists(ckpt_path):
                torch.save(model.state_dict(), ckpt_path)
            # Always print the path line (the grid driver parses this); harmless if
            # the reward-best already saved the same path.
            print(f"Best model saved at step {step} with accuracy: {best_reward:.4f}, path: {ckpt_path}")
            print(f"[train] best combined-reward EMA = {best_reward:.4f}, checkpoint: {ckpt_path}")

        bar.close()

    # Start training
    train(model, train_dataloader, validation_dataloader, evaluation_dataloader, tokenizer, optimizer, epochs=args.epochs, eval_interval=args.eval_interval, start_epoch=resume_epoch, start_step=resume_step, best_acc_init=resume_best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to generate adversarial tokens")
    parser.add_argument('--atker_path', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--target_path', type=str, required=True, help='target model path')
    parser.add_argument('--save_to_path', type=str, required=True, help='target model path')
    parser.add_argument('--len_doc_max', type=int, default=512, help='max length of document')
    parser.add_argument('--atk_what', type=str, default='doc', choices=['prefix', 'doc'], help='attack strategy: "doc" (modify document tokens). "prefix" is not implemented.')
    parser.add_argument('--num_doc_masks', type=int, default=10,  help='number of tokens to mask in document')
    parser.add_argument('--alpha', type=float, default=0.5, help='the weight on the adversarial obj.')
    parser.add_argument('--reward_type', type=str, default='A', choices=['A', 'B', 'C'],
                        help="adversarial-reward variant: A=naive prob-drop baseline (default, unchanged), "
                             "B=sigmoid-squashed σ(Δ), C=confidence-weighted w(x)·Δ. B/C add one guard "
                             "query per sample on the original source doc.")
    parser.add_argument('--epochs', type=int, default=10, help='training epochs (default 10 = full; use 1 for the screening sweep).')
    parser.add_argument('--eval_interval', type=int, default=100, help='steps between in-loop validation passes (each queries the guard); larger = faster screening.')
    parser.add_argument('--linear_head', type=str2bool, default=True, help='True: built-in MLM head (default, matches existing results). False: 3-layer FFN head.')
    parser.add_argument('--sim_scorer', type=str, default='use', help="similarity-reward backend: 'use' (default, USE) or 'embedding_api' (see similarity_scorer.py)")
    parser.add_argument('--server_url', type=str, default="http://localhost:8000/v1", help='8000 for llama guard 3 1B, 8001 for 8B')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')

    args = parser.parse_args()    

    # args = argparse.Namespace(
    #     atker_path='bert-base-uncased', # Example path
    #     target_path='temp',
    #     len_doc_max=512,
    #     num_doc_masks=10,
    #     save_to_path='/usa/taikun/rl-attack/1training/llama-guard-attacker',
    #     atk_what='doc',
    #     num_doc_masks=10,
    #     alpha=0.5)
    
    
    print_config(args)
    main(args)