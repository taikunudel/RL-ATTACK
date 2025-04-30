#!/usr/bin/env python
# coding: utf-8

# ## 1. Prepare Dataloader
# Load model directly
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from copy import deepcopy
import re
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertForMaskedLM, BertConfig

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import DataLoader, Subset, RandomSampler
torch.set_default_dtype(torch.float32)
# torch.autograd.set_detect_anomaly(True)
import sys
sys.path.append("..")
sys.path.append("/usa/taikun/07_transencoder")
# from llmrequest import requestNoDf
from utils import *

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

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

# def main():
def main(args):
    set_seed(42)
    
    importantTokensFile = args.importantTokensFile
    dataPath = args.dataPath
    dataName = args.dataName
    tgtModel = args.tgtModel
    attackerPath = args.attackerPath
    numsAttackedTokens = args.numsAttackedTokens
    maxLenDoc = args.maxLenDoc
    alpha = args.alpha
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.attacker_name == 'BERT_distill':
        attacker_path = 'distilbert/distilbert-base-uncased'
    elif args.attacker_name == 'BERT_large':
        attacker_path = 'google-bert/bert-large-uncased'
    else:
        attacker_path = 'bert-base-cased'
        
    # Load important tokens Json file
    with open(importantTokensFile, 'r') as f:
        importantTokens = json.load(f)

    importantKey = [list(d.keys()) for d in importantTokens]  
    importantKeys = [[int(v) for v in k] for k in importantKey]
    importantValues = [list(d.values()) for d in importantTokens]

    # # pad to the same length
    # padded_keys = [sublist + [-1] * (maxLenDoc - len(sublist)) for sublist in importantKeys]
    # padded_values = [sublist + [-0.1] * (maxLenDoc - len(sublist)) for sublist in importantValues]

    # Padding to the same length, maxLenDoc
    padded_keys = [sublist + [-1] * (maxLenDoc - len(sublist)) if len(sublist) < maxLenDoc else sublist[:maxLenDoc] for sublist in importantKeys]
    padded_values = [sublist + [-0.1] * (maxLenDoc - len(sublist)) if len(sublist) < maxLenDoc else sublist[:maxLenDoc] for sublist in importantValues]

    keys_tensor = torch.tensor(padded_keys)
    values_tensor = torch.tensor(padded_values)
    
    print("keys_tensor.shape", keys_tensor.shape)
    print("values_tensor.shape", values_tensor.shape)

    # texts_, labels_ = read_corpus(path = dataPath, clean=True, MR=False, lower=False, dataName=dataName)
    # labels_ = [label.strip() for label in labels_]
    # labels_ = labelProcessing(data=dataName, labels = labels_)

    tokenizer = AutoTokenizer.from_pretrained(attacker_path, max_length=128)
    vocabSize = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocabSize}")
    
    texts_, labels_ = read_corpus(path = dataPath, tokenizer = tokenizer, clean=True, MR=False, lower=False, dataName=dataName, dataPath=dataPath)
    
    texts_ = texts_[:keys_tensor.shape[0]]
    labels_ = labels_[:keys_tensor.shape[0]]

    print(f"Training Data Size: {keys_tensor.shape[0]}")
    
    # ### 1.2 Tokenize Dataset
    # examples is list[str]
    # max_length is int

    # def tokenize_function(examples, max_length=512):    
    #     return tokenizer(examples, padding='max_length', truncation=True, max_length=max_length)

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

    # tokenized_text = tokenize_function(texts_, tokenizer=tokenizer, max_length=512)
    # tokenized_text = tokenize_function(examples=texts_, tokenizer=tokenizer, max_length=512)
    tokenized_text = tokenize_function(examples=texts_,\
                                    dataName=dataName, tokenizer=tokenizer, max_length=512)

    # ### 1.3 Create Dataloader
    input_ids = torch.tensor(tokenized_text["input_ids"])
    attention_mask = torch.tensor(tokenized_text["attention_mask"])
    labels = torch.tensor(labels_)

    print(input_ids.shape)
    print(attention_mask.shape)
    print(labels.shape)
    print(keys_tensor.shape)
    print(values_tensor.shape)

    train_dataset = TensorDataset(input_ids, attention_mask, labels, keys_tensor, values_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=False)

    # # 2. Prepare Attacker
    # ### 2.1 Loading Attacker
    # Initialize the attacker
    # config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)
    # bertEncoder = BertForMaskedLM.from_pretrained(attacker_path, config=config)
    # for param in bertEncoder.parameters():
    #     param.requires_grad = False
    # generator = Generator(d_model=bertEncoder.config.hidden_size, vocab = vocabSize)
    # model = BertAttacker(bertEncoder, generator)

    # count_dict = {}

    criterion_quality = nn.CrossEntropyLoss(reduction='none', ignore_index = pad_token_id)
    criterion_adv = nn.CrossEntropyLoss(reduction='none', ignore_index = pad_token_id)
    compute_loss = CombLossCompute_nli(criterion_quality, criterion_adv, tgtModel, alpha)

    # # ### 2.3 Optimizer
    # # freeze paras except generator
    # generator_parameters = model.generator.parameters()
    # # for name, param in model.named_parameters():
    # #     if "generator" not in name:  # Adjust the condition based on your model structure
    # #         param.requires_grad = False

    # # optimizer = torch.optim.Adam(
    # # generator_parameters,  # Pass only generator parameters
    # # lr=0.01,
    # # betas=(0.9, 0.98),
    # # eps=1e-9
    # # )
    
    # config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)
    # model = BertForMaskedLM.from_pretrained(attacker_path, config=config)



    # Define the same custom classification head used during training
    class NonLinearHead(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.dense1 = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.ReLU()
            self.dense2 = nn.Linear(hidden_size, hidden_size)
            self.activation2 = nn.ReLU()
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.decoder = nn.Linear(hidden_size, vocab_size)

        def forward(self, hidden_states):
            x = self.activation(self.dense1(hidden_states))
            x = self.activation2(self.dense2(x))
            x = self.layer_norm(x)
            logits = self.decoder(x)
            return logits

    # Load configuration
    # attacker_path = "bert-base-cased"  # Change to your attacker path if needed
    config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)

    # Initialize model
    model = BertForMaskedLM.from_pretrained(attacker_path, config=config).to(device)

    # Replace the classification head with the custom one
    custom_head = NonLinearHead(config.hidden_size, config.vocab_size)
    model.cls = custom_head  # Replace standard head with custom one
    
        
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

    # # 3/ Train Attacker
    def train(model, dataloader, tokenizer, optimizer, epochs=10, eval_interval=5, seed=42):  
         
        eval_data = Subset(dataloader.dataset, range(100))
        train_data = Subset(dataloader.dataset, range(100, len(dataloader.dataset)))

        # Create new dataloaders with deterministic shuffling
        train_sampler = RandomSampler(train_data, replacement=False, generator=torch.Generator().manual_seed(seed))
        train_dataloader = DataLoader(train_data, batch_size=dataloader.batch_size, sampler=train_sampler)
        eval_dataloader = DataLoader(eval_data, batch_size=100, shuffle=False)
        
        model.to(device)
        model.train()
        # best_accuracy = 1.0
        best_loss = 10000.0
        step = 0
        for epoch in range(epochs):
            for batch in train_dataloader:
                step += 1
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                importantKeys = batch[3].tolist()
                importantValues = batch[4].tolist()
                importantTokens = [
                    {str(i): val for i, val in enumerate(vals) if keys[i] != -1}
                    for keys, vals in zip(importantKeys, importantValues)
                ]           
                
                optimizer.zero_grad()
                
                # nov 24 24
                # logits = model(input_ids, attention_mask) # [bS, maxDocLen, vcabSize]
                ## # maxInx = torch.argmax(logits, dim=-1) # tensor int64 [bS, maxDocLen]
                # probabilities = F.softmax(logits, dim=-1) # [bS, maxDocLen]
                
                logits = model(input_ids, attention_mask).logits # [bS, maxDocLen, vcabSize]
                probabilities = F.softmax(logits, dim=-1) # [bS, maxDocLen]
                
                # # # # Check for NaNs or Infs in probabilities and print their positions and values if any
                # maxInx = []
                # for pi in range(logits.shape[0]):
                #     mI = torch.multinomial(probabilities[pi], 1).squeeze(-1).tolist()
                #     maxInx.append(mI)
                # maxInx = torch.tensor(maxInx).to(device)
                
                maxInx = []
                for pi in range(logits.shape[0]): # [bs, maxDocLen, vocabsize]
                    # Get the top 20 indices
                    p_list = []
                    for position in probabilities[pi]: # [vocabsize] in [maxDocLen, vocabsize]
                        topk_values, topk_indices = torch.topk(position, 20)  # [20]               
                        # Randomly select one index from the top 20
                        mI = random.choice(topk_indices.tolist()) # 
                        p_list.append(mI)
                    maxInx.append(p_list)
                maxInx = torch.tensor(maxInx).to(device)
                
                maxInxCut = [[t.item() for t in maxIx if t != pad_token_id][1:-1] for maxIx in maxInx]
                # maxInxCut = [row[1:-1] for row in maxInx.tolist()] # list [bS, maxDocLen-2]
                srcInx = deepcopy(input_ids)  # list 
                # srcInxCut = [row[1:-1] for row in srcInx]
                srcInxCut = [[t.item() for t in srcIx if t != pad_token_id][1:-1] for srcIx in srcInx]
                srcText = decode_batch(srcInxCut, tokenizer) # list of text [bS, docLen]

                
                
                # make copyText
                attackedInx = rankDicts(importantTokens, numsAttackedTokens)
                
                print(attackedInx)
                # breakpoint()
                # copyInx = makeCopyInx(srcInxCut, maxInxCut, attackedInx)
                copuInxCut = makeCopyInx(srcInxCut, maxInxCut, attackedInx)
                copyText = decode_batch(copuInxCut, tokenizer)
                
                src_tuples, copy_tuples = [], []
                for tuple_inx in range(len(input_ids)):  
                    decoded_premise, decoded_hypothesis = make_premise_hypothesis_from_indices(input_ids[tuple_inx], tokenizer)
                    src_tuple = (decoded_premise, decoded_hypothesis)
                    src_tuples.append(src_tuple)
                    
                    copy_decoded_premise, copy_decoded_hypothesis = make_premise_hypothesis_from_indices([101] + copuInxCut[tuple_inx] + [102], tokenizer)
                    copy_tuple = (copy_decoded_premise, copy_decoded_hypothesis)
                    copy_tuples.append(copy_tuple)
                
                loss, advLoss, quaLoss, avgLLM, advRewards, sucFlag = \
                    compute_loss(modelOut=logits, srcText=srcText, copyText=copyText, maxInx = maxInx,\
                        docLen=maxInx.size(-1), important_tokens=attackedInx,\
                            src_tuple=src_tuples, copy_tuple=copy_tuples)
                
                if not sucFlag:
                    print(f"Skip Step {step}, LLM Failed.")
                    continue

                loss.backward()
                optimizer.step()
                
                srcTextHl = makeHighlightText(srcInxCut, attackedInx, tokenizer, attacker_name=args.attacker_name)
                copyTextHl = makeHighlightText(copuInxCut, attackedInx, tokenizer, attacker_name=args.attacker_name)
                predBinary, predData = query(srcText, tgtModel)
                copyPredBinary, copyPredProb = query(copyText, tgtModel)
                if step % 1 == 0:
                    print(f"step: {step}")
                    print(f"source    : {srcTextHl[0]}")
                    print(f"generate  : {copyTextHl[0]}")
                    # print(f"impTokens : {attackedInx[0]}")
                    print(f"srcProb   : {[f'{num:.4f}' for num in predData[0]]}")
                    print(f"genProb   : {[f'{num:.4f}' for num in copyPredProb[0]]}")
                
                # chekcing adversarias
                equal_elements = [0 if a == b else 1 for a, b in zip(copyPredBinary, predBinary)]
                for l in range(len(equal_elements)):
                    if equal_elements[l] == 1:
                        print(f"source    : {srcTextHl[l]}")
                        print(f"generate  : {copyTextHl[l]}")
                        print(f"srcPred   : {predBinary[l]}")
                        print(f"srcProb   : {[f'{num:.4f}' for num in predData[l]]}")
                        print(f"genPred   : {copyPredBinary[l]}")
                        print(f"genProb   : {[f'{num:.4f}' for num in copyPredProb[l]]}")
                        
                lr = optimizer.param_groups[0]["lr"]
                # elapsed = time.time() - start
                elapsed = 0

                # if (step % 1 == 0) and (loss > 0):
                if (step % 1 == 0):
                    accuracy = 1 - (sum(equal_elements) / len(srcInx))
                    print(
                        (
                            "step: %4d |Loss: %4.4f "
                            + "|BertAcc: %4.2f |AdvLoss: %4.4f |QuaLoss: %4.4f"
                            + "| AdvRew: %4.4f | QuaRew: %4.2f |LR: %5.1e")
                        % (step, loss, accuracy, advLoss, quaLoss, advRewards, avgLLM, lr))
                
                if loss < best_loss:
                    best_loss = loss
                
                if (step <= 300) and (step % 30 == 0):
                    model_path = f"{attackerPath}attacker_{dataName}_{alpha}_{epoch}_{step}_{best_loss:.4f}.pth"
                    torch.save(model.state_dict(), model_path)
                    print("step", step)
                    print(f"Model saved with loss: {loss:.4f}, {best_loss:.4f} at path: {model_path}")
                
                if (step < 10000) and (step % 100 == 0):
                    model_path = f"{attackerPath}attacker_{dataName}_{alpha}_{epoch}_{step}_{best_loss:.4f}.pth"
                    torch.save(model.state_dict(), model_path)
                    print("step", step)
                    print(f"Model saved with loss: {loss:.4f}, {best_loss:.4f} at path: {model_path}")
                    
                    # # if best_loss < 0.1:
                    # model_path = f"{attackerPath}attacker_{dataName}_{alpha}_{epoch}_{step}_{best_loss:.4f}.pth"
                    # torch.save(model.state_dict(), model_path)
                    # print(f"Model saved with loss: {best_loss:.4f} at path: {model_path}")
            
    train(model=model, dataloader=train_dataloader, optimizer=optimizer, tokenizer=tokenizer, epochs=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # common
    parser.add_argument('--dataName', type=str, required=True, help='yelp yelp_val imdb imdb_val')
    parser.add_argument('--dataPath', type=str, required=True, help='path to train/val datasets')
    parser.add_argument('--importantTokensFile', type=str, required=True, help='path to save & read preprocessed important tokens file')
    parser.add_argument('--tgtModel', type=str, required=True, help='target model path')
    parser.add_argument('--attackerPath', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--numsAttackedTokens', type=int, required=True, help='numbers of the prec of desired attacked tokens')
    parser.add_argument('--maxLenDoc', type=int, default=512, help='max length of document')  # Default value set to 512
    parser.add_argument('--attacker_name', type=str, help='attacker base') 
    
    # training
    parser.add_argument('--alpha', type=float, required=True, help='weight for adv training loss')

    args = parser.parse_args()    
    print_config(args)
    main(args)
    
    # main()