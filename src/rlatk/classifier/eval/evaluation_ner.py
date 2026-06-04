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
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# Set the logging level to 'error' to suppress warnings
logging.set_verbosity_error()

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
sys.path.append("/usa/taikun/rl-attack")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from llmrequest import requestNoDf
from rlatk.classifier.utils import *

def main():
    # dataPath = args.dataPath
    
    # importantTokensFile = args.importantTokensFile
    importantTokensFile = '/usa/taikun/rl-attack/0dataProcessing/tokens_ner.json'
    
    # Load attacked tokens Json file
    with open(importantTokensFile, 'r') as f:
        importantceData = json.load(f)
    
    
    # numCandidatesEachToken = args.numCandidatesEachToken
    numCandidatesEachToken = 1000
    # ifPrintAttackProcess = args.ifPrintAttackProcess
    # evaluationPartition = args.evaluationPartition
    
    # startFromSample = args.startFromSample
    startFromSample = 0
    # mode = args.mode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # attacker_name = 'dslim/bert-base-NER'
    # attacker_name = 'bert-base-cased'
    attacker_name = 'distilbert/distilbert-base-cased'

    # Load the CoNLL-2003 dataset
    dataset = load_dataset("conll2003")
    data_labels = dataset["train"].features["ner_tags"].feature.names
    print("Categories:", data_labels)
    print("Number of categories:", len(data_labels))

    # Load the model and tokenizer
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(attacker_name)
    vocabSize = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocabSize}")
    
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    model_labels = model.config.id2label    
    
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

    # tokenized_text = tokenize_function(examples=texts_,\
    #                                 dataName=dataName, tokenizer=tokenizer, max_length=512)

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

    # # # Initialize the attacker
    # # config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)
    # config = BertConfig.from_pretrained(attacker_name, output_hidden_states=True)
    # # if attackerName == 'BERTMaskedLMRandom':
    # # bertEncoder = BertForMaskedLM.from_pretrained(attacker_path, config=config)
    # bertEncoder = BertForMaskedLM.from_pretrained(attacker_name, config=config)
    # generator = Generator(d_model=bertEncoder.config.hidden_size, vocab = vocabSize)
    # testModel = BertAttacker(bertEncoder, generator).to(device).eval()
        
    # # breakpoint()
    
    # # # BERT LINEAR ATTACK
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BertConfig.from_pretrained(attacker_name, output_hidden_states=True)
    bertEncoder = BertForMaskedLM.from_pretrained(attacker_name, config=config).eval()
    for param in bertEncoder.parameters():
        param.requires_grad = False
    generator = Generator(d_model=bertEncoder.config.hidden_size, vocab=vocabSize)
    testModel = BertAttacker(bertEncoder, generator).to(device)
    
    # # bertlinear
    # attackerFile = "/usa/taikun/rl-attack/1training/ner/0209bertlinear/attacker_conll2003_1_28_3973_0.5918_0.4650.pth"
    attackerFile = "/usa/taikun/rl-attack/1training/ner/distilattacker_conll2003_1_0_1094_0.3481_0.4694.pth"
    state_dict = torch.load(attackerFile, map_location=device)
    testModel.load_state_dict(state_dict, strict=True)
        
    # # BERT NON-LINEAR ATTACK
    # attackerFile = "/usa/taikun/rl-attack/1training/ner/0209bertnonlinear/attacker_conll2003_1_9_1370_0.6596_0.5395.pth"
    # attackerFile = "/usa/taikun/rl-attack/1training/ner/0209bertnonlinear/attacker_conll2003_1_50_6987_0.6967_0.5554.pth"
    
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
        
    # # Load configurationxx
    # attacker_path = "bert-base-cased"  # Change to your attacker path if needed
    # config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)

    # # Initialize model
    # testModel = BertForMaskedLM.from_pretrained(attacker_path, config=config).to(device)

    # # Replace the classification head with the custom one
    # custom_head = NonLinearHead(config.hidden_size, config.vocab_size)
    # testModel.cls = custom_head  # Replace standard head with custom one
    
    # # Freeze all layers except the MLM head
    # for name, param in testModel.named_parameters():
    #     if 'cls' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    testModel.to(device)

    def attack(dataset_split, attacker):
        sourceToks, advToks, atkcandidates = [], [], []
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
            
        
        # for batch in valid_dataloader: # batch size is 1
        #  for i in tqdm(range(num_samples)):
        sourceDocs, advDocs, srcLabel, advLabel, targetInx, atkFlag, attackPatterns, cnts, docLen = [], [], [], [], [], [], [], [], []
        for i in tqdm(range(len(dataset["test"]))):
        # for i in tqdm(range(740,741)):
            example = dataset_split[i]
            tokens = example["tokens"]
            ground_truth = example["ner_tags"]
            references = [data_labels[tag] for tag in ground_truth]
            
            tokenized_input = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True).to(device)
            word_ids = tokenized_input.word_ids()
            tokenized_ground_truth, tokenized_refs = [], []
            for p in range(len(tokenized_input['input_ids'][0])):
                if word_ids[p] is not None:
                    tokenized_ground_truth.append(ground_truth[word_ids[p]])
                    tokenized_refs.append(references[word_ids[p]])
            
            importantTokens = [importantceData[i]]
            # total_from += 1
            # if total_from < startFromSample: # start from  startFromSample
            #     continue
            # print(total_from," th document.")
            
            total += 1
            # input_ids = batch[0].to(device)
            # attention_mask = batch[1].to(device)
            # label = batch[2].to(device)
            # label = batch[2]
            
            
            
            # srcInx = deepcopy(input_ids)  # list 
            # srcInxCut = [[t.item() for t in srcIx if t != pad_token_id][1:-1] for srcIx in srcInx]
            # srcText = decode_batch(srcInxCut, tokenizer) # list of text [bS, docLen]
            # srcText = decode_batch_nli(srcInxCut, tokenizer) # list of text [bS, docLen]
            # print("src : ", srcText[0])
            
            srcText = ' '.join(tokens)
            tokenized_important_toks = {}
            
            for ids in range(len(word_ids[1:-1])):
                word_id_ = word_ids[1:-1][ids]
                tokenized_important_toks[str(ids)] = importantTokens[0][str(word_id_)]
            
            attackedInx = rank_dict_ner(dicts=[tokenized_important_toks])
            attentionIdx = [[str(key) for key in aix.keys()] for aix in attackedInx]
            
            #print(attentionIdx)
            #print(attentionIdx)
            
            
            # srcTextHl = makeHighlightText(srcInxCut, attentionIdx, tokenizer)
            # print("stgt: ", srcTextHl[0])
            srcTextHl = srcText
            print("stgt: ", srcTextHl)
            
            maxInx, candidates = sample_decode_ner(attacker=testModel, tokenizer=tokenizer, tokens=tokenized_input, \
                attackedIndex=attackedInx, numCandidatesEachToken=numCandidatesEachToken,\
                unwanted_words=unwanted_words) # maxInx is buggy
            
            # replace machanism
            # decoded_premise, decoded_hypothesis = make_premise_hypothesis_from_indices(input_ids[0], tokenizer)
            
            # input = [(decoded_premise, decoded_hypothesis)]
            # sLabels, sProbs = query(input, tgtModel)  # [bs=1, trueDocLen]
            # sProbs = torch.tensor(sProbs).to(device) # [tensor(list(1,2))]
            
            # with torch.no_grad():
            #     outputs = model(**tokenized_input).logits
            #     prob = torch.softmax(outputs, dim=-1)[:, 1:-1, :] # excluding start and end None tokens
                
            #     true_label_prob = prob[0, torch.arange(len(tokenized_ground_truth)), tokenized_ground_truth] # [bs=1, doc_len (excluding start and end None tokens), ner_nums] -> [doc_len - 2, excluding start and end None tokens]
            #     # pred_l = torch.argmax(outputs.logits, dim=2).squeeze().tolist() # [bs=1, doc_len, nums_ner] -> [doc_len]
            #     # pred_prob = pred_l[tokenized_ground_truth] # [scale]
            
            if i % 100 == 0:
                results = {
                "sourceDocs": sourceDocs,
                "advDocs": advDocs,
                "sourceToks": sourceToks,
                "advToks": advToks,
                "srcLabel": srcLabel,
                "advLabel": advLabel,
                "atkcandidates": atkcandidates,
                "targetInx": targetInx,
                "atkFlag": atkFlag,
                "attackPatterns": attackPatterns,
                "docLen": docLen,
                "cnts": cnts}
                filename = 'coll2003_trained_distilllinear_0209.json'
                # Save the dictionary to a JSON file
                with open(filename, 'w') as json_file:
                    json.dump(results, json_file, indent=4)
                print(f"Results saved to {filename}")
            
            sourceDocs, sourceToks, advDocs, advToks, srcLabel, advLabel,\
                atkcandidates, targetInx, atkFlag, attackPatterns, docLen, cnts\
                    = doc_replace_ner(sample_num=i, subtokens=word_ids, tokens=tokens, src_labels=tokenized_ground_truth, atk_inx=attentionIdx,\
                        candidates=candidates, tokenizer=tokenizer, tgt_model=tgt_model, \
                            sourceDocs=sourceDocs, advDocs=advDocs, srcLabel=srcLabel, advLabel=advLabel, targetInx=targetInx, atkFlag=atkFlag, attackPatterns=attackPatterns, docLen=docLen, cnts=cnts, \
                        data_labels = data_labels, model_labels = model_labels, sourceToks=sourceToks, advToks=advToks, atkcandidates=atkcandidates)
            
            # sourceDocs, advDocs, srcLabel, advLabel, targetInx, atkFlag, attackPatterns, docLen, cnts = doc_replace_ner(sample_num=i, subtokens=word_ids, tokens=tokens,
            #     src_labels=tokenized_ground_truth, atk_inx=attentionIdx, candidates=candidates, tokenizer=tokenizer, tgt_model=model_name, \
            #         sourceDocs=sourceDocs, advDocs=advDocs, srcLabel=srcLabel, advLabel=advLabel, targetInx=targetInx, atkFlag=atkFlag, attackPatterns=attackPatterns, docLen=docLen, cnts=cnts, \
            #             data_labels = data_labels, model_labels = model_labels)
        
        results = {
        "sourceDocs": sourceDocs,
        "advDocs": advDocs,
        "sourceToks": sourceToks,
        "advToks": advToks,
        "srcLabel": srcLabel,
        "advLabel": advLabel,
        "atkcandidates": atkcandidates,
        "targetInx": targetInx,
        "atkFlag": atkFlag,
        "attackPatterns": attackPatterns,
        "docLen": docLen,
        "cnts": cnts}
        filename = 'coll2003_trained_distilllinear_0209.json'
        # Save the dictionary to a JSON file
        with open(filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Results saved to {filename}")
        return sourceDocs, advDocs, srcLabel, advLabel, targetInx, atkFlag, attackPatterns, docLen
            # print(res)


            # advInxCut, cntQueries, AttackFlag, advPred, attPattern = doc_replace_ner(input=tokens, subtokens=word_ids, tokens=tokens,
            #     src_labels=tokenized_ground_truth, atk_inx=attentionIdx, candidates=candidates, tokenizer=tokenizer, tgt_model=model_name)

            # srcTextHl = makeHighlightText(srcInxCut, [[str(key) for key in attPattern.keys()]], tokenizer)
            # # advText = tokenizer.decode(advInxCut, skip_special_tokens=True)
            # # advText = tokenizer.decode(advInxCut, skip_special_tokens=False)
            # advText = decode_batch_nli([advInxCut], tokenizer)
            # advTextHl = makeHighlightText([advInxCut], [[str(key) for key in attPattern.keys()]], tokenizer)

            # print('advInxCut', advInxCut)
            # print('advText', advText)
            # # statistics
            # totalQueries += cntQueries
            # avgQueries = (totalQueries / total) if total != 0 else totalQueries
            # attPatterns.append(attPattern)

            # sourcePredProb.append(sProbs.tolist()[0])
            # advPredProb.append(advPred.tolist())

            # docLen.append(len(srcInxCut[0]))

            # # create lists to be returned
            # sourceDocs.append(srcText[0])
            # sourceLabels.append(sProbs.tolist())
            # advDocs.append(advText[0])
            # srcHighlightedDocs.append(srcTextHl[0])

            # if AttackFlag:
            #     advCnt += 1
            #     advHighlightedDocs.append(advTextHl[0])
            #     attSucFlags.append(1)
                
            #     targetInx.append([str(key) for key in attPattern.keys()])
            #     # tgtTokens = [[input_ids[i][int(key)].item() for key in targetInx[i]] for i in range(len(targetInx))]

            #     tgtTokens, tgtTokensList = [], []
            #     for i in range(len(targetInx[-1])):
            #         inx = int(targetInx[-1][i])
            #         tgtTokensList.append(str(input_ids[-1][1:-1][inx].item()))
            #     tgtTokens.append(tgtTokensList)

            #     # tgtTokens = [input_ids[int(key)] for key in targetInx]
            #     targetTokens.append(tokenizer.convert_ids_to_tokens(tgtTokens[0]))
            #     toTokens.append(tokenizer.convert_ids_to_tokens([str(value) for value in attPattern.values()]))
            #     print("srcTxt: ", srcHighlightedDocs[-1])
            #     print("advTxt: ", advHighlightedDocs[-1])
            #     print("srcTxt: ", sourceDocs[-1])
            #     print("advTxt: ", advDocs[-1])
            #     print("tLabel: ", label.item() )
            #     print("targetTokens", targetTokens[-1])
            #     print("toTokens", toTokens[-1])
            #     print("sProbs: ", sourcePredProb[-1])
            #     print("aProbs: ", advPredProb[-1])
            # else:
            #     advHighlightedDocs.append('Unattacked')
            #     attSucFlags.append(0)

            #     targetInx.append([])
            #     targetTokens.append([])
            #     toTokens.append([])
            # print(f"attack suc rate {round((advCnt/total),4)*100} %")
            # print(f"average # of queries: {totalQueries/total}.")

            # if (total+1) % 1 == 0:
            #     data = {
            #         'srcDocshl': srcHighlightedDocs,
            #         'advDocshl': advHighlightedDocs,
            #         'srcLabel': sourceLabels,
            #         'sourceDocs': sourceDocs,
            #         'advDocs': advDocs,
            #         'srcPredProb': sourcePredProb,
            #         'advPredProb': advPredProb,
            #         'avgQueries': avgQueries,
            #         'targetInx': targetInx,
            #         'targetTokens': targetTokens,
            #         'toTokens': toTokens,
            #         'attackSuccess Flags': attSucFlags,
            #         'AattackPatterns': attPatterns,
            #         'docLen': docLen
            #     }
            #     total_docs_lens += sum(docLen)
            #     total_perturbed_tokens_nums += sum([len(t) for t in toTokens])
            #     print("all perturbed tokens : ", round(total_perturbed_tokens_nums, 2))
            #     print("total token lens     : ", round(total_docs_lens, 2))
            #     print("avg perturbed tokens : ", round(total_perturbed_tokens_nums/total, 2))
            #     print("avg perturbation rate: ", round(total_perturbed_tokens_nums/total_docs_lens, 4))

            #     # Save the dictionary to a JSON file
            #     with open(evaluationPartition, 'w') as json_file:
            #         json.dump(data, json_file, indent=4)
            #     print("saved at sample ",total, " as file ", evaluationPartition)
                
        # ****
        # ruturn adversarial sample and prediction
        # return sourceDocs, sourceLabels, advDocs, srcHighlightedDocs, advHighlightedDocs,\
        #     total, advCnt, avgQueries, attSucFlags, attPatterns, sourcePredProb, advPredProb,\
        #         targetInx, targetTokens, toTokens, docLen
        # return res


    # sourceDocs, sourceLabels, advDocs, srcHighlightedDocs, advHighlightedDocs,\
    #         total, advCnt, avgQueries, attSucFlags, attPatterns, sourcePredProb, advPredProb,\
    #             targetInx, targetTokens, toTokens, docLen =\
    #                     attack(valid_dataloader=eval_dataloader, attacker=testModel)
    
    # # ****
    # sourceDocs, sourceLabels, advDocs, srcHighlightedDocs, advHighlightedDocs,\
    #         total, advCnt, avgQueries, attSucFlags, attPatterns, sourcePredProb, advPredProb,\
    #             targetInx, targetTokens, toTokens, docLen =\
    #                     attack(dataset["test"], attacker=testModel)
    sourceDocs, advDocs, srcLabel, advLabel, targetInx, atkFlag, attackPatterns, docLen =  attack(dataset["test"], attacker=testModel)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="")
    # common
    # parser.add_argument('--dataName', type=str, required=True, help='yelp yelp_val imdb imdb_val')
    # parser.add_argument('--mode', type=str, required=True, help='classification or nli')
    # parser.add_argument('--dataPath', type=str, required=True, help='path to train/val datasets')
    # parser.add_argument('--importantTokensFile', type=str, required=True, help='path to save & read preprocessed important tokens file')
   #  parser.add_argument('--tgtModel', type=str, required=True, help='target model path')
    # parser.add_argument('--tgt_model_name', type=str, required=True, help='target model name')
    #parser.add_argument('--numsAttackedTokens', type=float, required=True, help='numbers of the prec of desired attacked tokens')
    #parser.add_argument('--maxLenDoc', type=int, default=512, help='max length of document')  # Default value set to 512
   #  parser.add_argument('--attacker_name', type=str, help='attacker base') 
    # attacking
    #parser.add_argument('--attackerName', type=str, required=True, help='name of the attacker')
    #parser.add_argument('--attackerFile', type=str, required=True, help='file of the attacker model')
    # parser.add_argument('--numCandidatesEachToken', type=int, required=True, help='total nums of the candidates each token')
    #parser.add_argument('--numsMaxCandidates', type=int, required=True, help='max numbers of the candidates each token')
    # parser.add_argument('--ifPrintAttackProcess', type=bool, required=True, help='whether to print the attack process')
    #parser.add_argument('--startFromSample', type=int, default=1, help='startFromSample')
    # parser.add_argument('--evaluationPartition', type=str, required=True, help='path to save the evaluation results json')
    
    # args = parser.parse_args()
    # main(args)
    main()

