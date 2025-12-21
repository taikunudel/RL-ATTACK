#!/usr/bin/env python
# coding: utf-8

# ## 1. Prepare Dataloader
# Load model directly
import argparse
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
import os
import sys
# sys.path.append("..")
sys.path.append("/usa/taikun/07_transencoder")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from llmrequest import requestNoDf
from utils import *

# def main():
def main(args):
    # device = torch.device(get_free_gpu())
    # print(f"Running on: {device}")

    dataPath = args.dataPath
    importantTokensFile = args.importantTokensFile
    dataName = args.dataName
    attackerName = args.attackerName
    tgt_model_name = args.tgt_model_name
    tgtModel = args.tgtModel
    attackerFile = args.attackerFile
    numsAttackedTokens = args.numsAttackedTokens # percent
    numsMaxCandidates = args.numsMaxCandidates
    numCandidatesEachToken = args.numCandidatesEachToken
    maxLenDoc = args.maxLenDoc
    ifPrintAttackProcess = args.ifPrintAttackProcess
    evaluationPartition = args.evaluationPartition
    
    startFromSample = args.startFromSample
    mode = args.mode
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

    if args.attacker_name == 'BERT_distill':
        attacker_path = 'distilbert/distilbert-base-uncased'
    elif args.attacker_name == 'BERT_large':
        attacker_path = 'google-bert/bert-large-uncased'
    else:
        attacker_path = 'bert-base-cased'
    
    print('attackerName', attackerName)
    print('attacker_path', attacker_path)
        
    ### 1.1 Read Dataset
    with open(importantTokensFile, 'r') as f:
        importantceData = json.load(f) 
    numsProcessedSamples = len(importantceData)    
    print("numsProcessedSamples", numsProcessedSamples)
    
    # examples is list[str]
    # max_length is int
    tokenizer = AutoTokenizer.from_pretrained(attacker_path)
    vocabSize = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocabSize}")

    texts_, labels_ = read_corpus(path = dataPath, tokenizer = tokenizer, clean=True, MR=False,\
        lower=False, dataName=dataName, dataPath=dataPath, tgtModel_name=tgt_model_name)
 
    texts_ = texts_[:numsProcessedSamples]
    labels_ = labels_[:numsProcessedSamples]
    print(f"Evaluation Data Size: {len(texts_)}")

    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id
    # Print the special token IDs
    print(f"CLS token ID: {cls_token_id}")
    print(f"SEP token ID: {sep_token_id}")
    print(f"PAD token ID: {pad_token_id}")
    print(f"MASK token ID: {mask_token_id}")
    print(f"UNK token ID: {unk_token_id}")

    # Get special token IDs filtered at generating candidates
    unwanted_words = [cls_token_id, sep_token_id, pad_token_id, mask_token_id, unk_token_id]
    # get all tokens with "unused" in target_tokenizer
    for token, index in tokenizer.vocab.items():
            if "unused" in token:
                unwanted_words.append(index)

    tokenized_text = tokenize_function(examples=texts_,\
                                    dataName=dataName, tokenizer=tokenizer, max_length=512)

    # # ### 1.2 Create Dataloader
    input_ids = torch.tensor(tokenized_text["input_ids"]).to(device)
    attention_mask = torch.tensor(tokenized_text["attention_mask"]).to(device)
    labels = torch.tensor(labels_).to(device)
    print(input_ids.shape)
    print(attention_mask.shape)
    print(labels.shape)
    print(numsProcessedSamples)

    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    eval_dataloader = deepcopy(train_dataloader)

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Also, if you're using other libraries like numpy or random, set their seeds as well
        import numpy as np
        np.random.seed(seed)
        import random
        random.seed(seed)
        
    # Define the custom non-linear head
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

    # Initialize the attacker
    config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)
    if attackerName == 'BERTMaskedLMRandom':
        bertEncoder = BertForMaskedLM.from_pretrained(attacker_path, config=config)
        generator = Generator(d_model=bertEncoder.config.hidden_size, vocab = vocabSize)
        testModel = BertAttacker(bertEncoder, generator).eval().to(device)
    elif attackerName == 'BERTFineTuned':
        print("attacker base: bert-base-cased ")
        bertEncoder = BertForMaskedLM.from_pretrained(attacker_path, config=config)
        # print("attacker base: bert-base-uncased ")
        # bertEncoder = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config)
        generator = Generator(d_model=bertEncoder.config.hidden_size, vocab = vocabSize)
        testModel = BertAttacker(bertEncoder, generator).eval().to(device)
        state_dict = torch.load(attackerFile)
        testModel.load_state_dict(state_dict, strict=True)
    elif attackerName == 'BERT_distill':
        config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)
        testModel = BertForMaskedLM.from_pretrained(attacker_path, config=config).eval().to(device)
        state_dict = torch.load(attackerFile)
        testModel.load_state_dict(state_dict, strict=True)
    elif attackerName == 'BERT_nonlinear': 
        testModel = BertForMaskedLM.from_pretrained(attacker_path, config=config).eval()
        custom_head = NonLinearHead(config.hidden_size, config.vocab_size)
        testModel.cls = custom_head
        state_dict = torch.load(attackerFile, map_location=device)
        testModel.load_state_dict(state_dict, strict=True)
        testModel = testModel.to(device)
    else:
    #if attackerName == 'BERTMaskedLM':
        # testModel = BertForMaskedLM.from_pretrained("bert-base-cased").eval().to(device)
        # print("attacker base: bert-base-cased ")
        testModel = BertForMaskedLM.from_pretrained("bert-base-uncased").eval().to(device)
        print("testModel: bert-base-uncased ")
        # testModel = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased").eval().to(device)
        # print("testModel: google-bert/bert-base-uncased")
        # testModel = BertForMaskedLM.from_pretrained("google-bert/bert-base-cased").eval().to(device)
        # print("testModel: google-bert/bert-base-cased")
    print("attacker device: ", next(testModel.parameters()).device)

    def attack(valid_dataloader, attacker):
        sourceDocs, sourcePredProb, sourceLabels, advDocs, advPredProb = [], [], [], [], []
        srcHighlightedDocs, advHighlightedDocs = [], []
        targetInx, targetTokens, toTokens = [], [], []
        docLen = []
        total, total_from, advCnt = 0.0, 0.0, 0.0
        totalQueries = 0.0
        attSucFlags = []
        attPatterns = []
        total_perturbed_tokens_nums = 0
        total_docs_lens = 0
            
        for batch in valid_dataloader: # batch size is 1
            importantTokens = [importantceData[int(total_from)]]
            total_from += 1
            if total_from < startFromSample: # start from  startFromSample
                continue
            print(total_from," th document.")
            
            total += 1
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            # label = batch[2].to(device)
            label = batch[2]
            
            srcInx = deepcopy(input_ids)  # list 
            srcInxCut = [[t.item() for t in srcIx if t != pad_token_id][1:-1] for srcIx in srcInx]
            # srcText = decode_batch(srcInxCut, tokenizer) # list of text [bS, docLen]
            srcText = decode_batch_nli(srcInxCut, tokenizer) # list of text [bS, docLen]
            # print("src : ", srcText[0])
            
            attackedInx = rankDicts(dicts=importantTokens, numsMaxCandidates=numsMaxCandidates)
            attentionIdx = [[str(key) for key in aix.keys()] for aix in attackedInx]
            
            srcTextHl = makeHighlightText(srcInxCut, attentionIdx, tokenizer)
            print("stgt: ", srcTextHl[0])

        # generate candidates
            maxInx, candidates = sample_decode(attacker=testModel, 
                                               attackerName=attackerName, input_ids=input_ids,\
                                                  attention_mask=attention_mask, attackedIndex=attackedInx,\
                                                    numCandidatesEachToken=numCandidatesEachToken,\
                                                          unwanted_words=unwanted_words) # maxInx is buggy
        # replace machanism
            decoded_premise, decoded_hypothesis = make_premise_hypothesis_from_indices(input_ids[0], tokenizer)
            
            input = [(decoded_premise, decoded_hypothesis)]
            sLabels, sProbs = query(input, tgtModel)  # st [bs=1, trueDocLen]
            sProbs = torch.tensor(sProbs).to(device) # [tensor(list(1,2))]
        
            advInxCut, cntQueries, AttackFlag, advPred, attPattern = doc_replace_nli(doc=srcInxCut, oriProb=sProbs,\
                oriLabel=label, attentionIdx=attentionIdx,\
                        candidates=candidates, printCandidates=ifPrintAttackProcess,\
                            tokenizer=tokenizer, tgtModel=tgtModel, mode=mode)

            srcTextHl = makeHighlightText(srcInxCut, [[str(key) for key in attPattern.keys()]], tokenizer)
            # advText = tokenizer.decode(advInxCut, skip_special_tokens=True)
            # advText = tokenizer.decode(advInxCut, skip_special_tokens=False)
            advText = decode_batch_nli([advInxCut], tokenizer)
            advTextHl = makeHighlightText([advInxCut], [[str(key) for key in attPattern.keys()]], tokenizer)

            print('advInxCut', advInxCut)
            print('advText', advText)
            # statistics
            totalQueries += cntQueries
            avgQueries = (totalQueries / total) if total != 0 else totalQueries
            attPatterns.append(attPattern)

            sourcePredProb.append(sProbs.tolist()[0])
            advPredProb.append(advPred.tolist())

            docLen.append(len(srcInxCut[0]))

            # create lists to be returned
            sourceDocs.append(srcText[0])
            sourceLabels.append(sProbs.tolist())
            advDocs.append(advText[0])
            srcHighlightedDocs.append(srcTextHl[0])

            if AttackFlag:
                advCnt += 1
                advHighlightedDocs.append(advTextHl[0])
                attSucFlags.append(1)
                
                targetInx.append([str(key) for key in attPattern.keys()])
                # tgtTokens = [[input_ids[i][int(key)].item() for key in targetInx[i]] for i in range(len(targetInx))]

                tgtTokens, tgtTokensList = [], []
                for i in range(len(targetInx[-1])):
                    inx = int(targetInx[-1][i])
                    tgtTokensList.append(str(input_ids[-1][1:-1][inx].item()))
                tgtTokens.append(tgtTokensList)

                # tgtTokens = [input_ids[int(key)] for key in targetInx]
                targetTokens.append(tokenizer.convert_ids_to_tokens(tgtTokens[0]))
                toTokens.append(tokenizer.convert_ids_to_tokens([str(value) for value in attPattern.values()]))
                print("srcTxt: ", srcHighlightedDocs[-1])
                print("advTxt: ", advHighlightedDocs[-1])
                print("srcTxt: ", sourceDocs[-1])
                print("advTxt: ", advDocs[-1])
                print("tLabel: ", label.item() )
                print("targetTokens", targetTokens[-1])
                print("toTokens", toTokens[-1])
                print("sProbs: ", sourcePredProb[-1])
                print("aProbs: ", advPredProb[-1])
            else:
                advHighlightedDocs.append('Unattacked')
                attSucFlags.append(0)

                targetInx.append([])
                targetTokens.append([])
                toTokens.append([])
            print(f"attack suc rate {round((advCnt/total),4)*100} %")
            print(f"average # of queries: {totalQueries/total}.")

            if (total+1) % 1 == 0:
                data = {
                    'srcDocshl': srcHighlightedDocs,
                    'advDocshl': advHighlightedDocs,
                    'srcLabel': sourceLabels,
                    'sourceDocs': sourceDocs,
                    'advDocs': advDocs,
                    'srcPredProb': sourcePredProb,
                    'advPredProb': advPredProb,
                    'avgQueries': avgQueries,
                    'targetInx': targetInx,
                    'targetTokens': targetTokens,
                    'toTokens': toTokens,
                    'attackSuccess Flags': attSucFlags,
                    'AattackPatterns': attPatterns,
                    'docLen': docLen
                }
                total_docs_lens += sum(docLen)
                total_perturbed_tokens_nums += sum([len(t) for t in toTokens])
                print("all perturbed tokens : ", round(total_perturbed_tokens_nums, 2))
                print("total token lens     : ", round(total_docs_lens, 2))
                print("avg perturbed tokens : ", round(total_perturbed_tokens_nums/total, 2))
                print("avg perturbation rate: ", round(total_perturbed_tokens_nums/total_docs_lens, 4))

                # Save the dictionary to a JSON file
                with open(evaluationPartition, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                print("saved at sample ",total, " as file ", evaluationPartition)
                

        # ruturn adversarial sample and prediction
        return sourceDocs, sourceLabels, advDocs, srcHighlightedDocs, advHighlightedDocs,\
            total, advCnt, avgQueries, attSucFlags, attPatterns, sourcePredProb, advPredProb,\
                targetInx, targetTokens, toTokens, docLen


    sourceDocs, sourceLabels, advDocs, srcHighlightedDocs, advHighlightedDocs,\
            total, advCnt, avgQueries, attSucFlags, attPatterns, sourcePredProb, advPredProb,\
                targetInx, targetTokens, toTokens, docLen =\
                        attack(valid_dataloader=eval_dataloader, attacker=testModel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # common
    parser.add_argument('--dataName', type=str, required=True, help='yelp yelp_val imdb imdb_val')
    parser.add_argument('--mode', type=str, required=True, help='classification or nli')
    parser.add_argument('--dataPath', type=str, required=True, help='path to train/val datasets')
    parser.add_argument('--importantTokensFile', type=str, required=True, help='path to save & read preprocessed important tokens file')
    parser.add_argument('--tgtModel', type=str, required=True, help='target model path')
    parser.add_argument('--tgt_model_name', type=str, required=True, help='target model name')
    parser.add_argument('--numsAttackedTokens', type=float, required=True, help='numbers of the prec of desired attacked tokens')
    parser.add_argument('--maxLenDoc', type=int, default=512, help='max length of document')  # Default value set to 512
    parser.add_argument('--attacker_name', type=str, help='attacker base') 
    # attacking
    parser.add_argument('--attackerName', type=str, required=True, help='name of the attacker')
    parser.add_argument('--attackerFile', type=str, required=True, help='file of the attacker model')
    parser.add_argument('--numCandidatesEachToken', type=int, required=True, help='total nums of the candidates each token')
    parser.add_argument('--numsMaxCandidates', type=int, required=True, help='max numbers of the candidates each token')
    parser.add_argument('--ifPrintAttackProcess', type=bool, required=True, help='whether to print the attack process')
    parser.add_argument('--startFromSample', type=int, default=1, help='startFromSample')
    parser.add_argument('--evaluationPartition', type=str, required=True, help='path to save the evaluation results json')
    
    args = parser.parse_args()
    main(args)
    # main()

