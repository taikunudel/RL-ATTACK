#!/usr/bin/env python
# coding: utf-8

# ## 1. Prepare Dataloader
# Load model directly
from functools import cache
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from copy import deepcopy
import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertForMaskedLM,
    BertConfig,
    DataCollatorForTokenClassification
)
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import DataLoader, Subset, RandomSampler
torch.set_default_dtype(torch.float32)
from names_dataset import NameDataset
import random

def print_config(args):
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Generator(nn.Module):
    def __init__(self, d_model, vocab, seed=42):  # d_model = input features, vocab = output features
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # Linear layer
        if seed is not None:
            self.set_seed_(seed)
        self.init_weights()  # Initialize the weights with Gaussian distribution

    def set_seed_(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    def init_weights(self):
        # Initialize the weights with Gaussian N(0, 1)
        nn.init.normal_(self.proj.weight, mean=0, std=1)  # N(0, 1) for weights
        print("Initialize the weights with Gaussian N(0, 1)")

        # # Initialize the weights with xavier_uniform_
        # torch.nn.init.xavier_uniform_(self.proj.weight)
        # print("Initialize the weights with xavier_uniform_")

        # # LeCun Normal Initialization (Manual)
        # nn.init.normal_(self.proj.weight, mean=0, std=1 / np.sqrt(self.proj.weight))
        # print("LeCun Normal Initialization (Manual)")

        # # Orthogonal Initialization
        # nn.init.orthogonal_(self.proj.weight)
        # print("Orthogonal Initialization")

        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)  # Set bias to zero, or you can also use normal_ if needed

    def forward(self, x):
        return self.proj(x)


class BertAttacker(nn.Module):
    def __init__(self, bertEncoder, generator, seed=None):
        super(BertAttacker, self).__init__()
        self.bert = bertEncoder
        self.generator = generator
        if seed is not None:
            set_seed(seed)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = outputs.hidden_states[-1]
        logits = self.generator(hidden_state) + 1e-3
        return logits


def main():
# def main(args):
    set_seed(42)
    epochs = 100
    nd_reward = 1
    nums_sample_per_tok = 20
    
    importantTokensFile = '/usa/taikun/07_transencoder/0dataProcessing/tokens_ner_train.json'
    tgt_model = 'dslim/bert-base-NER'
    
    # attacker_model = 'bert-base-cased'
    attacker_model = 'distilbert/distilbert-base-uncased'
    
    
    # Load attacked tokens Json file
    with open(importantTokensFile, 'r') as f:
        importantTokens = json.load(f)
    importantTokens = [json.dumps(line) for line in importantTokens]

    tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    tokenizer = AutoTokenizer.from_pretrained(attacker_model, max_length=512)
    vocabSize = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocabSize}")
    
    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id
    specialTokens = [cls_token_id, sep_token_id, pad_token_id]
    # Print the special token IDs
    print(f"CLS token ID: {cls_token_id}")
    print(f"SEP token ID: {sep_token_id}")
    print(f"PAD token ID: {pad_token_id}")
    print(f"MASK token ID: {mask_token_id}")
    print(f"UNK token ID: {unk_token_id}")
    
    # 1) Load the CoNLL-2003 dataset
    dataset = load_dataset("conll2003")
    label_list = dataset["train"].features["ner_tags"].feature.names
    print("Categories:", label_list)
    print("Number of categories:", len(label_list))
        
    train_dataset = dataset["train"]
    train_dataset = train_dataset.add_column("atk_idx", importantTokens)

    def filter_empty(example):
        return any(v > 0 for k, v in json.loads(example["atk_idx"]).items())

    train_dataset = train_dataset.filter(filter_empty)
    print(train_dataset)

    # 2) Tokenize
    def tokenize(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=128
        )

        token2subword_batch = []
        tokensize_batch = []
        for i in range(len(tokenized_inputs.input_ids)):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            token2subword = []
            tokensize = []
            current_word = None
            for i, word_idx in enumerate(word_ids):
                if word_idx is None:
                    pass
                elif word_idx != current_word:
                    token2subword.append(i)
                    tokensize.append(1)
                    current_word = word_idx
                else:
                    tokensize[-1] += 1
            token2subword_batch.append(token2subword)
            tokensize_batch.append(tokensize)
        tokenized_inputs["token2subword"] = token2subword_batch
        tokenized_inputs["tokensize"] = tokensize_batch
        return tokenized_inputs

    tokenized_train = train_dataset.map(tokenize, batched=True)

    tokenized_train.set_format(
        type=None,  # do not convert everything automatically
        columns=list(tokenized_train.column_names)
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def custom_collate_fn(features):
        numeric_keys = ["input_ids", "attention_mask"]
        numeric_features = []
        text_keys = ["tokens", "ner_tags", "token2subword", "tokensize", "atk_idx"]
        text_features = {}

        for sample in features:
            numeric_part = {}
            # Collect numeric columns
            for k in numeric_keys:
                numeric_part[k] = sample[k]

            # Collect non-numeric columns
            for k, v in sample.items():
                if k in text_keys:
                    # We'll store them in a dict of lists
                    if k not in text_features:
                        text_features[k] = []
                    text_features[k].append(v)

            numeric_features.append(numeric_part)

        numeric_batch = data_collator(numeric_features)
        final_batch = dict(numeric_batch)
        final_batch.update(text_features)  # add the raw columns (id, tokens, etc.)
        return final_batch

    train_dataloader = DataLoader(
        tokenized_train,
        # batch_size=32,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # linear model
    config = BertConfig.from_pretrained(attacker_model, output_hidden_states=True)
    bertEncoder = BertForMaskedLM.from_pretrained(attacker_model, config=config).eval()
    for param in bertEncoder.parameters():
        param.requires_grad = False
    generator = Generator(d_model=bertEncoder.config.hidden_size, vocab=vocabSize)
    model = BertAttacker(bertEncoder, generator).to(device)
    
    # Print shapes of all parameters
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
        
    # non-linear model
    # Define the same custom classification head used during training
    # class NonLinearHead(nn.Module):
    #     def __init__(self, hidden_size, vocab_size):
    #         super().__init__()
    #         self.dense1 = nn.Linear(hidden_size, hidden_size)
    #         self.activation = nn.ReLU()
    #         self.dense2 = nn.Linear(hidden_size, hidden_size)
    #         self.activation2 = nn.ReLU()
    #         self.layer_norm = nn.LayerNorm(hidden_size)
    #         self.decoder = nn.Linear(hidden_size, vocab_size)

    #     def forward(self, hidden_states):
    #         x = self.activation(self.dense1(hidden_states))
    #         x = self.activation2(self.dense2(x))
    #         x = self.layer_norm(x)
    #         logits = self.decoder(x)
    #         return logits

    # # Load configuration
    # attacker_path = "bert-base-cased"  # Change to your attacker path if needed
    # config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)

    # # Initialize model
    # model = BertForMaskedLM.from_pretrained(attacker_path, config=config).to(device)

    # # Replace the classification head with the custom one
    # custom_head = NonLinearHead(config.hidden_size, config.vocab_size)
    # model.cls = custom_head  # Replace standard head with custom one
    
    #     # Freeze all layers except the MLM head
    # for name, param in model.named_parameters():
    #     if 'cls' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    model.to(device)
            
    tgt_model = AutoModelForTokenClassification.from_pretrained(tgt_model).to(device).eval()
    nd = NameDataset()

    @cache
    def name_exist(name):
        nonlocal nd
        name_check = nd.search(name)
        return name_check['first_name'] or name_check['last_name']

    grad_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    print(grad_params)
    optimizer = torch.optim.Adam(
        grad_params.values(),
        lr=0.01,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # 3/ Train Attacker
    # model_lables = {0: 'O', 1: 'B-MISC', 2: 'I-MISC', 3: 'B-PER', 4: 'I-PER', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-LOC', 8: 'I-LOC'}
    PER_ID = 3

    avg_loss, avg_reward = None, None
    avg_adv_reward, avg_nd_reward = None, None
    avg_time = None
    
    def avg_update(old_v, new_v):
        if old_v is not None:
            return old_v * 0.99 + 0.01 * new_v
        else:
            return new_v
    
    step = 0
    for epoch in range(epochs):
        for batch in train_dataloader:
            start_time = time.time()

            step += 1
            input_ids = batch['input_ids'].to(model.bert.device)
            attention_mask = batch['attention_mask'].to(model.bert.device)
            importantTokens = [json.loads(line) for line in batch["atk_idx"]]
            token2subwords = batch["token2subword"]
            tokensizes = batch["tokensize"]
            
            optimizer.zero_grad()
            with torch.no_grad():
                action_probs = F.softmax(model(input_ids, attention_mask), dim=-1) # [bS, maxDocLen, vocab_size]
                # action_probs = F.softmax(model(input_ids, attention_mask).logits, dim=-1) # [bS, maxDocLen, vocab_size]

                # Decode token IDs back to text
                decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                # print(decoded_texts)  # Check if it makes sense
                re_tokenized = tokenizer_ner(decoded_texts, padding="max_length", truncation=True, return_tensors="pt")

                # Extract properly formatted inputs
                copy_ids = re_tokenized["input_ids"].to(device)
                copy_masks = re_tokenized["attention_mask"].to(device)

                #ner_prob = F.softmax(tgt_model(input_ids, attention_mask).logits, dim=-1)
                ner_prob = F.softmax(tgt_model(copy_ids, copy_masks).logits, dim=-1)
            
            '''`
            for each sample
            for each per token
            get its sampling
            make copy doc
            
            '''
            src_ids, copy_ids, copy_masks = [], [], []
            src_doc, copy_doc = [], []
            batch_indices, seq_indices, vocab_indices = [], [], []
            src_prob, copy_nd_reward = [], []
            for ai in range(len(batch['input_ids'])): # [bs, maxDocLen, vocabsize]
                atk_inx = [int(k) for k, v in importantTokens[ai].items() if v > 0]
                subword_first = [token2subwords[ai][idx] for idx in atk_inx]
                subword_end = [token2subwords[ai][idx] + tokensizes[ai][idx] - 1 for idx in atk_inx]
                for bi, bi_end in zip(subword_first, subword_end):
                    src_subword_ids = input_ids[ai].clone().detach() # [maxDocLen, vocabsize]
                    src_attention_mask = attention_mask[ai].clone().detach()
                    distribution = action_probs[ai][bi].clone().detach() # [vocabsize]
                    token_prob = ner_prob[ai][bi].clone().detach() # [numLabel]
                    sampled_inx = torch.multinomial(distribution, num_samples=nums_sample_per_tok).tolist()
                    original_id = src_subword_ids[bi]
                    for sampled_id in sampled_inx:
                        # if sampled_id == original_id:
                        #     continue
                        
                        copy_subword_ids = src_subword_ids.clone().detach()
                        copy_subword_ids[bi] = sampled_id

                        copy_word = tokenizer.decode(copy_subword_ids[bi:bi_end+1], skip_special_tokens=True)
                        if name_exist(copy_word):
                            copy_nd_reward.append(nd_reward)
                        else:
                            copy_nd_reward.append(0)
                        
                        src_ids.append(src_subword_ids)
                        src_doc.append(tokenizer.decode(src_subword_ids, skip_special_tokens=True))
                        copy_ids.append(copy_subword_ids)
                        copy_masks.append(src_attention_mask)
                        copy_doc.append(tokenizer.decode(copy_subword_ids, skip_special_tokens=True))
                        batch_indices.append(ai)
                        seq_indices.append(bi)
                        vocab_indices.append(sampled_id)
                        src_prob.append(token_prob)
            
            if len(src_ids) == 0:
                print('skip update as no sample')
                continue
            
            batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=model.bert.device)
            seq_indices = torch.tensor(seq_indices, dtype=torch.long, device=model.bert.device)
            vocab_indices = torch.tensor(vocab_indices, dtype=torch.long, device=model.bert.device)
            copy_nd_reward = torch.tensor(copy_nd_reward, dtype=torch.float32, device=model.bert.device)

            src_prob = torch.stack(src_prob)
            copy_ids = torch.stack(copy_ids)
            copy_masks = torch.stack(copy_masks)

            # print('src doc : ', src_doc[0])
            # print('copy doc: ', copy_doc[0])
            
            # Decode token IDs back to text
            decoded_texts = tokenizer.batch_decode(copy_ids, skip_special_tokens=True)
            # print(decoded_texts)  # Check if it makes sense
            re_tokenized = tokenizer_ner(decoded_texts, padding="max_length", truncation=True, return_tensors="pt")

            # Extract properly formatted inputs
            copy_ids = re_tokenized["input_ids"].to(copy_ids.device)
            copy_masks = re_tokenized["attention_mask"].to(copy_masks.device)

            with torch.no_grad():
                copy_ner_prob = torch.softmax(tgt_model(copy_ids, copy_masks).logits, dim=-1)
            
            copy_prob = copy_ner_prob[torch.arange(len(seq_indices), device=copy_ner_prob.device), seq_indices]
            probs_diff = src_prob - copy_prob
            adv_rewards = probs_diff[:, PER_ID]
            
            logits = model(input_ids, attention_mask)
            # logits = model(input_ids, attention_mask).logits # [bS, maxDocLen, vocab_size]
            logprobs = torch.log_softmax(logits, dim=-1)
            atk_logprob = logprobs[batch_indices, seq_indices, vocab_indices]
            loss = torch.mean(-1.0 * (adv_rewards + copy_nd_reward) * atk_logprob)
                   
            loss.backward()
            optimizer.step()

            avg_loss = avg_update(avg_loss, loss.item())
            avg_reward = avg_update(avg_reward, torch.mean(adv_rewards + copy_nd_reward).item())
            avg_adv_reward = avg_update(avg_adv_reward, torch.mean(adv_rewards).item())
            avg_nd_reward = avg_update(avg_nd_reward, torch.mean(copy_nd_reward).item())
            end_time = time.time()
            avg_time = avg_update(avg_time, end_time - start_time)
            print(f"Epoch: {epoch} Step:{step} Speed: {1/avg_time:.3f} iter/s Loss:{avg_loss:.3f} Reward:{avg_reward:.3f} Adv:{avg_adv_reward:.3f} ND:{avg_nd_reward:.3f}")
        model_path = f"/usa/taikun/07_transencoder/1training/ner/distilattacker_conll2003_1_{epoch}_{step}_{avg_adv_reward:.4f}_{avg_nd_reward:.4f}.pth"
        torch.save(model.state_dict(), model_path)
        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="")
    # # common
    # parser.add_argument('--dataName', type=str, required=True, help='yelp yelp_val imdb imdb_val')
    # parser.add_argument('--dataPath', type=str, required=True, help='path to train/val datasets')
    # parser.add_argument('--importantTokensFile', type=str, required=True, help='path to save & read preprocessed important tokens file')
    # parser.add_argument('--tgtModel', type=str, required=True, help='target model path')
    # parser.add_argument('--attackerPath', type=str, required=True, help='path folder to load attacker models')
    # parser.add_argument('--numsAttackedTokens', type=int, required=True, help='numbers of the prec of desired attacked tokens')
    # parser.add_argument('--maxLenDoc', type=int, default=512, help='max length of document')  # Default value set to 512
    # parser.add_argument('--attacker_name', type=str, help='attacker base') 
    
    # # training
    # parser.add_argument('--alpha', type=float, required=True, help='weight for adv training loss')

    # args = parser.parse_args()    
    # print_config(args)
    # main(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #torch.backends.cuda.matmul.allow_tf32 = False
    # torch.set_float32_matmul_precision("high")
    main()