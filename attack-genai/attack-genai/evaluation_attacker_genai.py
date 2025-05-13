#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import sys
import GPUtil
from collections import defaultdict # Added this import
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu
from itertools import product
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BertForMaskedLM, 
    BertConfig)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    TensorDataset, 
    Dataset, 
    DataLoader, 
    Subset, 
    RandomSampler)
import tensorflow as tf
# print("TensorFlow sees the following Physical GPUs:", tf.config.list_physical_devices('GPU'))
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:                          
#     tf.config.experimental.set_memory_growth(gpus[1], True)   # ❶
import tensorflow_hub as hub
from train_attacker_genai import *

# read data
# make 10 masks
# infer
# get the logits
# random fill in the masks
# print out and save all the attacked results
# test accuracy, USE, perturbation rate

# random vs trained
# trained on a single or both losses
# attack diffrent 

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

def save_lists_to_json(list_names: List[str], lists_to_zip: List[List[Any]], output_json_path: str):
    """
    Takes a list of list names and a list of lists, zips them together,
    creates a list of dictionaries, and saves it to a JSON file.

    Args:
        list_names: A list of strings, where each string is the key for a list's elements
                    in the output JSON dictionary. The order of names should correspond
                    to the order of lists in `lists_to_zip`.
        lists_to_zip: A list of lists to be zipped together. The number of lists
                      must match the number of names in `list_names`.
        output_json_path: The path to the JSON file where the results will be saved.

    Raises:
        ValueError: If the number of list names does not match the number of lists.
    """
    if len(list_names) != len(lists_to_zip):
        raise ValueError("The number of list names must match the number of lists to zip.")

    results: List[Dict[str, Any]] = []
    zipped_data = zip(*lists_to_zip)

    for item in zipped_data:
        result_dict: Dict[str, Any] = {}
        for i, name in enumerate(list_names):
            result_dict[name] = item[i]
        results.append(result_dict)

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to: {output_json_path}")

# def sample_multiple_without_replacement(logits: torch.Tensor, num_samples: int) -> torch.Tensor:
#     """
#     Samples multiple tokens without replacement for each position in the logits,
#     keeping the samples for each original token grouped.

#     Args:
#         logits: A tensor of shape (batch_size, num_tokens, vocab_size) representing the logits
#                 for each token position in the batch.
#         num_samples: The number of unique tokens to sample without replacement for each
#                      token position.

#     Returns:
#         A tensor of shape (batch_size, num_tokens, num_samples) containing the indices
#         of the sampled tokens.
#     """
#     batch_size, num_tokens, vocab_size = logits.shape
#     all_sampled_tokens = []

#     for i in range(num_tokens):
#         probs = torch.softmax(logits[:, i, :], dim=-1)
#         # Sample without replacement for each batch element
#         indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
#         all_sampled_tokens.append(indices)

#     # Stack the sampled tokens to create the desired shape
#     stacked_sampled_tokens = torch.stack(all_sampled_tokens, dim=1)
#     return stacked_sampled_tokens
    
# def get_top_k_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
#     """
#     Gets the top k token indices from the logits for each position.

#     Args:
#         logits: A tensor of shape (batch_size, num_tokens, vocab_size) representing the logits
#                 for each token position in the batch.
#         k: The number of top token indices to retrieve.

#     Returns:
#         A tensor of shape (batch_size, num_tokens, k) containing the indices
#         of the top k tokens.
#     """
#     batch_size, num_tokens, vocab_size = logits.shape
#     all_top_k_indices = []

#     for i in range(num_tokens):
#         # Get the top k values and indices for the current token position
#         topk_values, topk_indices = torch.topk(logits[:, i, :], k=k, dim=-1)
#         all_top_k_indices.append(topk_indices)

#     # Stack the top k indices to create the desired shape
#     stacked_top_k_indices = torch.stack(all_top_k_indices, dim=1)
#     return stacked_top_k_indices

def get_top_k_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Gets the top k token indices from the logits for each position,
    selecting from the top 100 candidates and then randomly choosing 5.

    Args:
        logits: A tensor of shape (batch_size, num_tokens, vocab_size) representing the logits
                for each token position in the batch.
        k: The number of token indices to retrieve (in this case, 5).

    Returns:
        A tensor of shape (batch_size, num_tokens, k) containing the indices
        of the randomly selected top k tokens.
    """
    batch_size, num_tokens, vocab_size = logits.shape
    all_top_k_indices = []

    top_n = 100  # Number of candidates to consider

    for i in range(num_tokens):
        # Get the top n values and indices for the current token position
        top_n_values, top_n_indices = torch.topk(logits[:, i, :], k=top_n, dim=-1)

        # Ensure random_indices is on the same device as top_n_indices
        random_indices = torch.randint(low=0, high=top_n, size=(batch_size, k), device=top_n_indices.device)
        
        # Use gather to get the actual indices
        top_k_indices = torch.gather(top_n_indices, dim=-1, index=random_indices)
        all_top_k_indices.append(top_k_indices)

    # Stack the top k indices to create the desired shape
    stacked_top_k_indices = torch.stack(all_top_k_indices, dim=1)
    return stacked_top_k_indices

def generate_token_sample_combinations_batched(sampled_tokens: torch.Tensor, sub_batch_size: int) -> torch.Tensor:
    """
    Generates combinations in smaller sub-batches to manage memory.

    Args:
        sampled_tokens: A tensor of shape (batch_size, num_tokens, num_samples).
        sub_batch_size: The number of documents to process for combinations at a time.

    Returns:
        A tensor of shape (batch_size, num_samples ** num_tokens, num_tokens)
        containing all combinations, processed in batches.
    """
    batch_size, num_tokens, num_samples = sampled_tokens.shape
    all_combinations = []

    for i in range(0, batch_size, sub_batch_size):
        sub_batch = sampled_tokens[i:i + sub_batch_size]
        combinations_sub_batch = []
        for b in range(sub_batch.shape[0]):
            token_samples = sub_batch[b].tolist()
            combinations = list(product(*token_samples))
            combinations_tensor = torch.tensor(combinations, dtype=torch.long)
            combinations_sub_batch.append(combinations_tensor)

        if combinations_sub_batch:
            all_combinations.append(torch.stack(combinations_sub_batch))

    if all_combinations:
        return torch.cat(all_combinations, dim=0)
    else:
        return torch.empty(0, num_samples ** num_tokens, num_tokens, dtype=torch.long)

def main(args):
    atker_path = args.atker_path
    target_path = args.target_path
    len_doc_max = args.len_doc_max
    prefix_length = args.prefix_length
    save_to_path = args.save_to_path
    samples_per_tok = args.samples_per_tok
    max_queries_per_doc = args.max_queries_per_doc
    atk_json_log = args.atk_json_log

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_gpu = GPUtil.getFirstAvailable(order='memoryFree', maxLoad=0.5, maxMemory=0.5)[0]
    torch.cuda.set_device(best_gpu)
    device = torch.device(f"cuda:{best_gpu}")
    print(f"Using GPU {best_gpu}")

    USE = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")
    # with tf.device(f"/GPU:{best_gpu}"):
    #     USE = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")

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

    train_dataloader, validation_dataloader, evaluation_dataloader = build_datasets(tokenizer=tokenizer, prefix_length=prefix_length, max_len=len_doc_max, seed=42)
    
    # Initialize attacker
    config = BertConfig.from_pretrained(atker_path, output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained(atker_path, config=config).to(device)
    
    # Load the trained model weights
    state_dict = torch.load(save_to_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)  # Load the weights

    # Freeze all layers.  This is evaluation, so typically we don't change model weights.
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.eval() # Set to eval mode
    
    def attack(evaluation_dataloader=evaluation_dataloader, attacker=model, eval_interval=100):
        model.eval()
        acc_metric = Accuracy(task="binary").to(device)
        all_results = []
        all_predicted_labels = []
        all_true_labels = []
        all_source_predicted_probs = []
        all_generated_predicted_probs = []
        all_source_documents = []
        all_generated_documents = []
        queries = []
        USEs = []
        pertubations = []
        with torch.no_grad():
            total_batches = len(evaluation_dataloader)
            bar = tqdm(total=total_batches, desc="Evaluating", unit="batch")            
            for batch_idx, batch in enumerate(evaluation_dataloader, start=1):
                curr_queries_per_doc = 0

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                src_doc = tokenizer.decode(input_ids[0],skip_special_tokens=True)
                prompt_, predictions_, probs_ = get_raw_logits.process_file(data=[src_doc]) 

                logits = model(input_ids, attention_mask).logits # [bS, maxDocLen, vcabSize]
                prefix_logits = logits[:, :prefix_length, :] # [batch_size, prefix_length, vocab_size]
                batch_size, num_prefix_tokens, vocab_size, prefix_labels, source_documents, generated_documents = logits_to_labels(input_ids, prefix_logits, tokenizer, pad_token_id, cls_token_id, sep_token_id, prefix_length=prefix_length)
                
                sampled_tokens = get_top_k_indices(logits=prefix_logits, k=samples_per_tok)

                all_combinations = generate_token_sample_combinations_batched(sampled_tokens, sub_batch_size=4)
                all_combinations = all_combinations.squeeze(0)

                # sampled_tokens = sample_multiple_without_replacement(logits=prefix_logits, num_samples=samples_per_tok)
                # all_combinations = generate_token_sample_combinations_batched(sampled_tokens, sub_batch_size=4)
                # all_combinations = all_combinations.squeeze(0)

                # prompt_, predicted_labels, probs_ = get_raw_logits.process_file(data=generated_documents)
                # true_labels = labels
                
                # predicted_labels_tensor = torch.tensor(predicted_labels).to(device)
                # predicted_probs_tensor = torch.tensor(probs_).to(device) # Get the probabilities

                # source_logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device)).logits
                # _, source_predicted_labels, source_probs = get_raw_logits.process_file(data=source_documents)
                # source_predicted_probs_tensor = torch.tensor(source_probs).to(device)
                
                # decode every possible comb
                all_text_combinations = tokenizer.batch_decode(all_combinations, skip_special_tokens=True)
                random.shuffle(all_text_combinations)
                # iteratively try each 
                
                atk_succ = False
                # for text in tqdm(all_text_combinations, desc="Processing Combinations"): # Added tqdm
                for text in all_text_combinations:
                    curr_queries_per_doc += 1
                    if curr_queries_per_doc >= max_queries_per_doc:
                        break
                    adv_doc = ' '.join([text, src_doc])
                    eva_prompt_, eval_predictions_, eval_probs_ = get_raw_logits.process_file(data=[adv_doc])  
                    if eval_predictions_[0] != labels[0]:
                        atk_succ = True              
                        break
                
                all_source_documents.extend(prompt_)
                all_generated_documents.extend(generated_documents)
                all_true_labels.extend(labels.cpu().numpy().tolist()) 
                if atk_succ:
                    all_predicted_labels.extend(eval_predictions_)
                else:
                    all_predicted_labels.extend(labels.cpu().numpy().tolist())
                all_source_predicted_probs.extend(probs_)
                all_generated_predicted_probs.extend(eval_probs_)
                
                all_true_labels_tensor = torch.tensor(all_true_labels).to(device)
                all_predicted_labels_tensor = torch.tensor(all_predicted_labels).to(device)
                
                acc_metric.update(all_predicted_labels_tensor, all_true_labels_tensor)
                current_accuracy = acc_metric.compute()

                queries.append(curr_queries_per_doc)
                USEs.append(getUSEcosSimilarity([all_source_documents[-1]], [all_generated_documents[-1]], USE)[0])
                pertubations.append(prefix_length / torch.sum(attention_mask[0]).item())

                bar.update(1)
                bar.set_postfix({
                    "avg_acc":      f"{current_accuracy:.4f}",
                    "avg_queries":  f"{np.mean(queries):.4f}",
                    "avg_pert":     f"{np.mean(pertubations)*100:.4f}",
                    "avg_USE":      f"{np.mean(USEs):.4f}",
                    "curr_queries": f"{queries[-1]:.4f}",
                    "curr_pert":    f"{pertubations[-1]*100:.4f}",
                    "curr_USE":     f"{USEs[-1]:.4f}"
                })

                if batch_idx % eval_interval == 0:
                    print(f"\n--- Step {batch_idx} ---")
                    print("Source Documents (last few):", all_source_documents[-1])
                    print("Generated Documents (last few):", all_generated_documents[-1])
                    print("True Labels (last few):", all_true_labels[-1])
                    print("Predicted Labels (last few):", all_predicted_labels[-1])
                    print("Source Predicted Probabilities:", round(all_source_predicted_probs[-1],4))
                    print("Generated Predicted Probabilities:", round(all_generated_predicted_probs[-1],4))
                
                list_names = ['src_doc', 'adv_doc', 'true_label', 'pred_label', 'queries', 'pertubations', 'USEs', 'true_prob', 'pred_prob']
                lists_to_zip = [all_source_documents, all_generated_documents, all_true_labels, all_predicted_labels, queries, pertubations, USEs, all_source_predicted_probs, all_generated_predicted_probs]
                save_lists_to_json(list_names=list_names, lists_to_zip=lists_to_zip, output_json_path=atk_json_log)
                
            bar.close()
        

    attack(evaluation_dataloader=evaluation_dataloader, attacker=model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--atker_path', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--target_path', type=str, required=True, help='target model path')
    parser.add_argument('--save_to_path', type=str, required=True, help='target model path')
    parser.add_argument('--len_doc_max', type=int, default=512, help='max length of document')  # Default value set to 512
    parser.add_argument('--prefix_length', type=int, default=10, help='')  # Default value set to 512
    parser.add_argument('--samples_per_tok', type=int, default=10, help='')  # Default value set to 512
    parser.add_argument('--max_queries_per_doc', type=int, default=50, help='')  # Default value set to 512
    parser.add_argument('--atk_json_log', type=str, default=10, help='')  # Default value set to 512


    # args = argparse.Namespace(
    #         atker_path='bert-base-uncased', # Example path
    #         target_path='temp',
    #         len_doc_max=512,
    #         prefix_length=10,
    #         save_to_path='/usa/taikun/07_transencoder/1training/llama-guard-attacker/attacker_llama-guard_4_5100_0.6450.pth',
    #         samples_per_tok=3,
    #         max_queries_per_doc=2,
    #         # atk_json_log = '/usa/taikun/07_transencoder/attack-genai/atk_greedy_topk_doc_log.json')
    #         atk_json_log = '/usa/taikun/07_transencoder/attack-genai/temp.json')
        
    args = parser.parse_args()    
    print_config(args)
    main(args)
