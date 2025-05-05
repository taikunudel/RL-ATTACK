#!/usr/bin/env python
# coding: utf-8

# ## 1. Prepare Dataloader
# Load model directly
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from copy import deepcopy
import random
import argparse
import json
import requests
from typing import List, Dict, Any
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
# from utils import *
import get_raw_logits

def print_config(args):
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def preprocess_function(examples, prefix_length, len_doc_max, tokenizer):
    mask_token = '[MASK]'
    prefix = ' '.join([mask_token] * prefix_length)
    inputs = [prefix + ' ' + doc for doc in examples['prompt']]
    tokenized_inputs = tokenizer(inputs, max_length=len_doc_max, truncation=True, padding='max_length')
    tokenized_inputs['labels'] = [1 if j else 0 for j in examples['jailbreak']]
    return tokenized_inputs

# Verify the class ratio in each split (optional)
def get_class_distribution(dataset):
    jailbreak_count = sum(dataset['labels'])
    regular_count = len(dataset) - jailbreak_count
    return {"jailbreak": jailbreak_count, "regular": regular_count, "total": len(dataset)}

def build_datasets(tokenizer, prefix_length, max_len, seed=42):
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
        lambda examples: preprocess_function(examples, prefix_length, max_len, tokenizer),
        batched=True
    )
    validation_data = validation_data.map(
        lambda examples: preprocess_function(examples, prefix_length, max_len, tokenizer),
        batched=True
    )
    evaluation_data = evaluation_data.map(
        lambda examples: preprocess_function(examples, prefix_length, max_len, tokenizer),
        batched=True
    )
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    validation_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    evaluation_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Define the batch size
    batch_size = 16  # You can adjust this as needed
    # Create DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
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

def logits_to_labels(input_ids, prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, prefix_length=10):
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
          
def main(args):
    atker_path = args.atker_path
    target_path = args.target_path
    len_doc_max = args.len_doc_max
    prefix_length = args.prefix_length
    save_to_path = args.save_to_path

    best_gpu = GPUtil.getFirstAvailable(order='memoryFree', maxLoad=0.5, maxMemory=0.5)[0]
    torch.cuda.set_device(best_gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    print(f"Using GPU {best_gpu}")

    tokenizer = AutoTokenizer.from_pretrained(atker_path, max_length=len_doc_max)
    vocab_size = tokenizer.vocab_size
    print(f'Attacker Base: {atker_path}')
    print(f"Vocabulary Size: {vocab_size}")

    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id

   #  print(f'Data: {data_path}')
    
    # ds = load_dataset("walledai/AdvBench", split="train")  # 520 rows
    # print(ds[0])                # see the first prompt‑target pair
    # prompts = ds["prompt"]      # grab just the instructions
    # targets = ds["target"]

    train_dataloader, validation_dataloader, evaluation_dataloader = build_datasets(tokenizer=tokenizer, prefix_length=prefix_length, max_len=len_doc_max, seed=42)
    
    # # 2. Prepare Attacker
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
    eps=1e-9)
    
    # # Load trained model weights
    # trained_model_path = "path_to_your_trained_model.pth"  # Update with actual path
    # state_dict = torch.load(trained_model_path, map_location=device)
    # model.load_state_dict(state_dict, strict=True)

    # def logits_to_labels(input_ids, prefix_logits):
    #     batch_size = prefix_logits.size(0)
    #     num_prefix_tokens = prefix_logits.size(1)
    #     vocab_size = prefix_logits.size(2)
    #     # Sample a token for each masked prefix position based on the probabilities
    #     prefix_probabilities = F.softmax(prefix_logits, dim=-1) # [batch_size, prefix_length, vocab_size]
    #     sampled_prefix_tokens = torch.multinomial(prefix_probabilities.view(-1, vocab_size), num_samples=1).view(batch_size, num_prefix_tokens) # [batch_size, prefix_length]
    #     prefix_labels = sampled_prefix_tokens.clone().detach()
    #     # Convert sampled prefix token IDs and original input IDs back to text for reward calculation
    #     generated_prefix_tokens_list = sampled_prefix_tokens.tolist()
    #     original_input_ids_list = input_ids.tolist()
    #     source_documents = []
    #     generated_documents = []

    #     for i in range(batch_size):            
    #         # Get the original document (excluding the prefix of [MASK] tokens)
    #         original_document_tokens = [token_id for idx, token_id in enumerate(original_input_ids_list[i]) if idx >= prefix_length and token_id != pad_token_id and token_id not in [cls_token_id, sep_token_id]]
    #         original_document = tokenizer.decode(original_document_tokens, skip_special_tokens=True)

    #         # Combine generated prefix and original document
    #         generated_prefix = tokenizer.decode(generated_prefix_tokens_list[i], skip_special_tokens=True)
    #         generated_document = generated_prefix + " " + original_document # You might want a different way to combine
    #         source_documents.append(original_document)
    #         generated_documents.append(generated_document)
    #     return batch_size, num_prefix_tokens, vocab_size, prefix_labels, source_documents, generated_documents
    
    # # 3/ Train Attacker
    def train(model, train_dataloader, validation_dataloader, evaluation_dataloader, tokenizer, optimizer, epochs, eval_interval=100, print_length=200):  
        model.train()
        best_acc, val_accuracy = float('inf'), float('inf')
        step = -1

        total_batches = len(train_dataloader)
        total_steps   = epochs * total_batches

        # single bar over all steps
        bar = tqdm(total=total_steps, desc="Training", unit="step")
        source_documents_all, generated_documents_all, labels_all, val_source_documents_all, val_generated_documents_all, val_labels_all = [], [], [], [], [], []
        for epoch in range(epochs):
            # for batch in train_dataloader:
            for batch_idx, batch in enumerate(train_dataloader, start=1):
                step += 1
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # based on the prompt, make surfix
                # know the real lengtth
                # how does BERT mask filling a document
                logits = model(input_ids, attention_mask).logits # [bS, maxDocLen, vcabSize]
                # prefix_logits = logits[:, :prefix_length, :]
                # prefix_probs = F.softmax(prefix_logits, dim=-1) # [bS, maxDocLen]
                # # src_from
                # # src_to
                prefix_logits = logits[:, :prefix_length, :] # [batch_size, prefix_length, vocab_size]
                batch_size, num_prefix_tokens, vocab_size, prefix_labels, source_documents, generated_documents = logits_to_labels(input_ids, prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, prefix_length=prefix_length)
                source_documents_all.extend(source_documents)
                generated_documents_all.extend(generated_documents)
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
                   
                # loss = F.cross_entropy(
                #     prefix_logits.view(-1, vocab_size),
                #     prefix_labels.view(-1),
                # )

                loss = torch.tensor(0.0, device=device)
                for i in range(batch_size):
                    batch_custom_loss = F.cross_entropy(
                        prefix_logits[i],  # Logits for the prefix of the i-th document [prefix_length, vocab_size]
                        prefix_labels[i],  # Sampled tokens for the prefix of the i-th document [prefix_length]
                    )
                    loss += batch_custom_loss * rewards[i]

                loss /= batch_size # Normalize by batch size

                loss.backward()
                optimizer.step()

                # update global step
                bar.update(1)
                bar.set_postfix({
                    "epoch": f"{epoch+1}/{epochs}",
                    "batch": f"{batch_idx}/{total_batches}",
                    "loss": f"{loss.item():.4f}"
                })   
                
                if step % eval_interval == 0:
                    model.eval()
                    val_loss_total = 0
                    val_batches = len(validation_dataloader)

                    # Initialize metrics
                    acc_metric = Accuracy(task="binary").to(device)
                    val_acc_metric = Accuracy(task="binary").to(device)
                    # precision_metric = Precision(average='macro').to(device) # 'macro' for multiclass
                    # recall_metric = Recall(average='macro').to(device)
                    # f1_metric = F1Score(average='macro').to(device)
                    with torch.no_grad():
                        for val_batch in validation_dataloader:
                            val_input_ids = val_batch['input_ids'].to(device)
                            val_attention_mask = val_batch['attention_mask'].to(device)
                            val_labels = val_batch['labels'].to(device)
                            val_logits = model(val_input_ids, val_attention_mask).logits
                            val_prefix_logits = val_logits[:, :prefix_length, :]
                            _, _, _, _, val_source_documents, val_generated_documents = logits_to_labels(val_input_ids, val_prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, prefix_length=prefix_length)
                            val_source_documents_all.extend(val_source_documents) 
                            val_generated_documents_all.extend(val_generated_documents)
                            val_prompt_, val_predictions_, val_probs_ = get_raw_logits.process_file(data=val_generated_documents)
                            val_labels_all.extend(val_labels)

                            # # Calculate validation loss (optional but good practice)
                            # batch_val_loss = torch.tensor(0.0, device=device)
                            # for i in range(len(val_labels)):
                            #     batch_val_loss += F.cross_entropy(
                            #         val_prefix_logits[i],
                            #         torch.zeros(prefix_length, dtype=torch.long, device=device)
                            #     )
                            # val_loss_total += batch_val_loss.item()

                            # Convert predictions to the correct format for metrics
                            val_predicted_classes = torch.tensor(val_predictions_).to(device)

                            # Update metrics
                            acc_metric.update(predicted_classes, labels)
                            val_acc_metric.update(val_predicted_classes, val_labels)
                            
                            # precision_metric.update(predicted_classes, val_labels)
                            # recall_metric.update(predicted_classes, val_labels)
                            # f1_metric.update(predicted_classes, val_labels)

                        # Compute metrics
                        accuracy = acc_metric.compute()
                        val_accuracy = val_acc_metric.compute()
                        # precision = precision_metric.compute()
                        # recall = recall_metric.compute()
                        # f1 = f1_metric.compute()
                        # avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0

                        # print(
                        #     f"Step: {step}, Training Loss: {loss.item():.4f}, "
                        #     f"Validation Loss: {avg_val_loss:.4f}, "
                        #     f"Validation Accuracy: {accuracy:.4f}, "
                        #     f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
                        # )

                        print(
                            f"Step: {step}, Training Loss: {loss.item():.4f}, "
                            f"Training Accuracy: {accuracy:.4f}, "
                            f"Validation Accuracy: {val_accuracy:.4f}, "
                        )
                        
                        train_idx = random.randint(0, len(source_documents_all) - 1)
                        val_idx = random.randint(0, len(val_source_documents_all) - 1)
                        print('train src: ', source_documents_all[train_idx][:print_length])
                        print('train gen: ', generated_documents_all[train_idx][:print_length])
                        print('train lab ', labels_all[train_idx])
                        print('val src: ', val_source_documents_all[val_idx][:print_length])
                        print('val gen: ', val_generated_documents_all[val_idx][:print_length])
                        print('val lab ', val_labels_all[train_idx])
                        source_documents_all, generated_documents_all, val_source_documents_all, val_generated_documents_all = [], [], [], []
                        # Reset metrics for the next validation stage
                        acc_metric.reset()
                        # precision_metric.reset()
                        # recall_metric.reset()
                        # f1_metric.reset()
                    model.train()
                    

                    # val_input_ids = validation_dataloader['input_ids'].to(device)
                    # val_attention_mask = validation_dataloader['attention_mask'].to(device)
                    # val_logits = model(val_input_ids, val_attention_mask).logits # [bS, maxDocLen, vcabSize]
                    # val_prefix_logits = val_logits[:, :prefix_length, :] # [batch_size, prefix_length, vocab_size]
                    # val_generated_documents = logits_to_labels(val_prefix_logits)
                    # val_prompt_, val_predictions_, _ = get_raw_logits.process_file(data=val_generated_documents)
                    # val_labels = validation_dataloader['labels']
                    # # calculate the accuracy

                    # print(f"Step: {step}, Training Loss: {loss.item():.8f}, ")
                    # # also print the accuracy in val dataset

                # if val_accuracy < 0.95 * best_acc and step >= 1000:
                if val_accuracy < best_acc:
                    best_acc = val_accuracy
                    model_path = f"{save_to_path}/attacker_llama-guard_{epoch}_{step}_{best_acc:.4f}.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"Model saved at step {step} with loss: {best_acc:.4f}, path: {model_path}")
        bar.close()  

    train(model, train_dataloader, validation_dataloader, evaluation_dataloader, tokenizer, optimizer, epochs=10, eval_interval=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--atker_path', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--target_path', type=str, required=True, help='target model path')
    parser.add_argument('--save_to_path', type=str, required=True, help='target model path')
    parser.add_argument('--len_doc_max', type=int, default=512, help='max length of document')  # Default value set to 512
    parser.add_argument('--prefix_length', type=int, default=10, help='')  # Default value set to 512

    args = argparse.Namespace(
            atker_path='bert-base-uncased', # Example path
            target_path='temp',
            len_doc_max=512,
            prefix_length=10,
            save_to_path='/usa/taikun/07_transencoder/1training/llama-guard-attacker/')
        
    # args = parser.parse_args()    
    print_config(args)
    main(args)