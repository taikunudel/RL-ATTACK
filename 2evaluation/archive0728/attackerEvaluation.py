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
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from llmrequest import requestNoDf
from utils import *

# def main():
def main(args):
    dataPath = args.dataPath
    importantTokensFile = args.importantTokensFile
    dataName = args.dataName
    attackerName = args.attackerName
    tgtModel = args.tgtModel
    attackerPath = args.attackerPath
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
    
        
    # # For MNLI matched Val Set
    # dataName='mnli_matched_val'
    # dataPath='/usa/taikun/07_transencoder/MNLI/mnli/mnli_matched.txt'
    # importantTokensFile='/usa/taikun/07_transencoder/0dataProcessing/mnli/NImportantTokens_mnli_matched_val_0.json_hypothesis.json_updated.json'
    # tgtModel='textattack/bert-base-uncased-MNLI'
    # attackerName='ALBERT'
    # attackerPath='None' 
    # numsAttackedTokens=1.0
    # maxLenDoc=512

    # alpha=-0.99

    # mode = 'nli'
    # attackerFile='None'
    # numCandidatesEachToken=150 # total numbers of desire attacked tokens
    # numsMaxCandidates=10
    # ifPrintAttackProcess=True
    # evaluationPartition="None"

    # startFromSample=0

    
    # # For MNLI mismatched Val Set
    # dataName='mnli_mismatched_val'
    # dataPath='/usa/taikun/07_transencoder/MNLI/mnli/mnli_mismatched.txt'
    # importantTokensFile='/usa/taikun/07_transencoder/0dataProcessing/mnli/NImportantTokens_mnli_mismatched_val_0.json_hypothesis.json_updated.json'
    # tgtModel="textattack/bert-base-uncased-MNLI"
    # attackerName='BERTMaskedLM'
    # attackerPath='None' 
    # numsAttackedTokens=0.7
    # maxLenDoc=512

    # alpha=0.7

    # mode = 'nli'
    # attackerFile='None'
    # numCandidatesEachToken=80 # total numbers of desire attacked tokens
    # numsMaxCandidates=8
    # ifPrintAttackProcess=True

    # startFromSample=15
    # mode = 'nli'
    # on='hypothesis'
    
    
    
    # # For Yelp Validation Set
    # dataName = 'yelp_val'
    # dataPath = '/usa/taikun/07_transencoder/yelp/yelp/yelp.txt'
    # importantTokensFile = '/usa/taikun/07_transencoder/0dataProcessing/NImportantTokens_yelp_val_0_updated.json'
    # tgtModel = "textattack/bert-base-uncased-yelp-polarity"
    # attackerName = 'BERTFineTuned'
    # attackerPath = '/usa/taikun/07_transencoder/yelp/attackerModels'
    # numsAttackedTokens = 0.6  # percent
    # maxLenDoc = 512

    # alpha = -0.1

    # attackerFile = '/usa/taikun/07_transencoder/yelp/attackerModelsattacker_yelp_0.7_6_133_0.6667.pth'
    # numCandidatesEachToken = 400  # total numbers of desired attacked tokens
    # numsMaxCandidates = 24
    # ifPrintAttackProcess = True
    # evaluationPartition = "yelp_val_temp.json"
    # startFromSample = 129
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # # For IMDB Validation Set
    # dataName = 'imdb_val'
    # dataPath = '/usa/taikun/07_transencoder/imdb/imdb.txt'
    # importantTokensFile = '/usa/taikun/07_transencoder/0dataProcessing/NImportantTokens_imdb_val_0_updated.json'
    # tgtModel = "textattack/bert-base-uncased-imdb"
    # attackerName = 'BERTFineTuned'
    # attackerPath = '/usa/taikun/07_transencoder/yelp/attackerModels'
    # numsAttackedTokens = 1.0
    # maxLenDoc = 512

    # alpha = -0.1

    # attackerFile = '/usa/taikun/07_transencoder/yelp/attackerModels/attacker_yelp_4_90_0.6875.pth'
    # numCandidatesEachToken = 800  # total numbers of desired attacked tokens
    # numsMaxCandidates = 48
    # ifPrintAttackProcess = True
    # evaluationPartition = "yelp_val_BERTFineTuned_MAX_CANDIDATES_EACH_TOKEN100.json"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    ### 1.1 Read Dataset

    # generate important tokens for training docs
    # input_ids_cuts = [[t.item() for t in input_id if t != pad_token_id][1:-1] for input_id in input_ids]
    # input_ids_tokens = [[tokenizer.convert_ids_to_tokens(t) for t in input_ids_cut] for input_ids_cut in input_ids_cuts]
    # importantTokens = getImportantTokens(input_ids_tokens)
    # keys_tensor = torch.tensor([list(d.keys()) for d in importantTokens])
    # values_tensor = torch.tensor([list(d.values()) for d in importantTokens])

    # # Save to a JSON file
    # with open('importantTokens.json', 'w') as f:
    #     json.dump(importantTokens, f)
    # print("saved")

    # # ====== 0706 ====
    # Load important tokens Json file
    # with open(importantTokensFile, 'r') as f:
    #     importantTokens = json.load(f) 
    
    # ========= 0707 added ========
    with open(importantTokensFile, 'r') as f:
        importantceData = json.load(f) 
    numsProcessedSamples = len(importantceData)    
    print("numsProcessedSamples", numsProcessedSamples)
    # ========= 0707 added ========
    
    # importantKey = [list(d.keys()) for d in importantTokens]
    # importantKeys = [[int(v) for v in k] for k in importantKey]
    # importantValues = [list(d.values()) for d in importantTokens]

    # # pad to the same length
    # padded_keys = [sublist + [-1] * (maxLenDoc - len(sublist)) for sublist in importantKeys]
    # padded_values = [sublist + [-0.1] * (maxLenDoc - len(sublist)) for sublist in importantValues]
    # # keys_tensor = torch.tensor(padded_keys)
    # # values_tensor = torch.tensor(padded_values)
    # keys_tensor = torch.tensor(padded_keys).to(device)
    # values_tensor = torch.tensor(padded_values).to(device)
    
    # # ====== 0706 ====
    
    
    # ### 1.2 Tokenize Dataset
    # examples is list[str]
    # max_length is int
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    vocabSize = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocabSize}")

    texts_, labels_ = read_corpus(path = dataPath, tokenizer = tokenizer, clean=True, MR=False, lower=False, dataName=dataName, dataPath=dataPath)
    # labels_ = [label.strip() for label in labels_]
    # labels_ = labelProcessing(data=dataName, labels = labels_)

    texts_ = texts_[:numsProcessedSamples]
    labels_ = labels_[:numsProcessedSamples]
    
    # texts_ = texts_[min(startFromSample, numsProcessedSamples):numsProcessedSamples]
    # labels_ = labels_[min(startFromSample, numsProcessedSamples):numsProcessedSamples]
    
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

    # ### 1.3 Create Dataloader
    # input_ids = torch.tensor(tokenized_text["input_ids"])
    # attention_mask = torch.tensor(tokenized_text["attention_mask"])
    # labels = torch.tensor(labels_)
    input_ids = torch.tensor(tokenized_text["input_ids"]).to(device)
    attention_mask = torch.tensor(tokenized_text["attention_mask"]).to(device)
    labels = torch.tensor(labels_).to(device)
    print(input_ids.shape)
    print(attention_mask.shape)
    print(labels.shape)
    # print(keys_tensor.shape)
    # print(values_tensor.shape)
    print(numsProcessedSamples)

    # train_dataset = TensorDataset(input_ids, attention_mask, labels, keys_tensor, values_tensor)
    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    eval_dataloader = deepcopy(train_dataloader)

    def makeImportanceMask(important_tokens, rows, cols, rewards):
        # importance_reward_mask = torch.zeros((rows, cols)).to(device)
        importance_reward_mask = torch.zeros((rows, cols), device=device)
        for inx, impTokens in enumerate(important_tokens):
            reward = rewards[inx]
            for tokenIds in impTokens.keys():
                importance_reward_mask[inx][int(tokenIds)] = reward
        return importance_reward_mask

    # # 3/ Train Attacker
    def makeHighlightText(DocInxCut, attackedInx):
        if not attackedInx:
            hlTextList = [tokenizer.decode(inx, skip_special_tokens=True) for inx in DocInxCut]
            return hlTextList
        
        hlTextList = []
        for inx in range(len(DocInxCut)):
            hlText = []
            for i in range(len(DocInxCut[inx])):
                Id = DocInxCut[inx][i]
                if str(i) in attackedInx[inx]:
                    highlighted_id = [164, 164, Id, 166, 166]
                else:
                    highlighted_id = [Id]
                hlText.extend(highlighted_id)
            hlTextList.append(tokenizer.decode(hlText, skip_special_tokens=True))
        return hlTextList

    def makeCopyInx(srcInxCut, maxInxCut, attackedInx):
        if not attackedInx:
            return srcInxCut
        
        copyInx = []
        for inx in range(len(srcInxCut)):
            modified_src = []
            for i in range(len(srcInxCut[inx])):
                if str(i) in attackedInx[inx]:
                    modified_src.append(maxInxCut[inx][i])
                else:
                    modified_src.append(srcInxCut[inx][i])
            copyInx.append(modified_src)
        return copyInx

    def getUSEcosSimilarity(srcDocs, copyDocs):
        # input 2 lists of documents 
        import tensorflow_hub as hub
        embed = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")
        USEcosinSimilarity = []
        sim_metric = torch.nn.CosineSimilarity(dim=1)
        for src, copy in zip(srcDocs, copyDocs):
            emb1, emb2 = embed([src, copy])["outputs"]
            # emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
            emb1, emb2 = torch.tensor(emb1.numpy()).to(device), torch.tensor(emb2.numpy()).to(device)
            srcEmb = torch.unsqueeze(emb1, dim=0).to(device) # [embSz] -> [1, embSz]
            advEmb = torch.unsqueeze(emb2, dim=0).to(device)
            es = sim_metric(srcEmb, advEmb)
            USEcosinSimilarity.append(es.item())
        return USEcosinSimilarity
    
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

    def evaluate(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(device)
                labels = batch[2].to(device)
                texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                predBinary, prob = query(texts, tgtModel) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy
    
    # Initialize the attacker
    config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
    if attackerName == 'BERTMaskedLMRandom':
        bertEncoder = BertForMaskedLM.from_pretrained("bert-base-cased", config=config)
        generator = Generator(d_model=bertEncoder.config.hidden_size, vocab = vocabSize)
        testModel = BertAttacker(bertEncoder, generator).eval().to(device)

    elif attackerName == 'BERTFineTuned':
        bertEncoder = BertForMaskedLM.from_pretrained("bert-base-cased", config=config)
        generator = Generator(d_model=bertEncoder.config.hidden_size, vocab = vocabSize)
        testModel = BertAttacker(bertEncoder, generator).eval().to(device)
        state_dict = torch.load(attackerFile)
        testModel.load_state_dict(state_dict, strict=True)
    else:
    #if attackerName == 'BERTMaskedLM':
        testModel = BertForMaskedLM.from_pretrained("bert-base-cased").eval().to(device)

    print("attacker device: ", next(testModel.parameters()).device)

    def attack(valid_dataloader, attacker):
        sourceDocs, sourcePredProb, sourceLabels, advDocs, advPredProb = [], [], [], [], []
        srcHighlightedDocs, advHighlightedDocs = [], []
        targetInx, targetTokens, toTokens = [], [], []
        docLen = []
        total, total_from, advCnt = 0.0, 0.0, 0.0
        totalQueries = 0.0
        llmCnt, llmScore = 0.0, 0.0
        attSucFlags = []
        attPatterns = []

        # read data
        # with open(importantTokensFile, 'r') as file:
        #     importantceData = json.load(file)
            
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
            # importantKeys = batch[3].tolist()
            # importantValues = batch[4].tolist()
            # importantTokens = [
            #         {str(i): val for i, val in enumerate(vals) if keys[i] != -1}
            #         for keys, vals in zip(importantKeys, importantValues)
            #     ]
            
            srcInx = deepcopy(input_ids)  # list 
            srcInxCut = [[t.item() for t in srcIx if t != pad_token_id][1:-1] for srcIx in srcInx]
            srcText = decode_batch(srcInxCut, tokenizer) # list of text [bS, docLen]
            # print("src : ", srcText[0])
            
            # 0707
            attackedInx = rankDicts(dicts=importantTokens, numsMaxCandidates=numsMaxCandidates)
            attentionIdx = [[str(key) for key in aix.keys()] for aix in attackedInx]
            
            # attentionIdx = [str(key) for d in attackedInx for key in d.keys()]
            srcTextHl = makeHighlightText(srcInxCut, attentionIdx)
            print("stgt: ", srcTextHl[0])

        # generate candidates
            maxInx, candidates = sample_decode(attacker=testModel, \
                                               attackerName=attackerName, input_ids=input_ids,\
                                                  attention_mask=attention_mask, attackedIndex=attackedInx,\
                                                    numCandidatesEachToken=numCandidatesEachToken,\
                                                          unwanted_words=unwanted_words) # maxInx is buggy

        # replace machanism
            sLabels, sProbs = query(srcText, tgtModel)  # st [bs=1, trueDocLen]
            sProbs = torch.tensor(sProbs).to(device) # [tensor(list(1,2))]
            # advInxCut, cntQueries, AttackFlag, advPred, attPattern = docReplace(doc=srcInxCut, oriProb=sProbs,\
            #     oriLabel=label, attentionIdx=attentionIdx,\
            #             candidates=candidates, printCandidates=ifPrintAttackProcess,\
            #                 tokenizer=tokenizer, tgtModel=tgtModel)
            
            advInxCut, cntQueries, AttackFlag, advPred, attPattern = docReplace(doc=srcInxCut, oriProb=sProbs,\
                oriLabel=label, attentionIdx=attentionIdx,\
                        candidates=candidates, printCandidates=ifPrintAttackProcess,\
                            tokenizer=tokenizer, tgtModel=tgtModel, mode=mode)
            
            srcTextHl = makeHighlightText(srcInxCut, [[str(key) for key in attPattern.keys()]])
            advText = tokenizer.decode(advInxCut, skip_special_tokens=True)
            advTextHl = makeHighlightText([advInxCut], [[str(key) for key in attPattern.keys()]])

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
            advDocs.append(advText)
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

            if (total+1) % 10 == 0:
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
    parser.add_argument('--attackerPath', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--numsAttackedTokens', type=float, required=True, help='numbers of the prec of desired attacked tokens')
    parser.add_argument('--maxLenDoc', type=int, default=512, help='max length of document')  # Default value set to 512
    # training
    parser.add_argument('--alpha', type=float, required=True, help='weight for adv training loss')
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

