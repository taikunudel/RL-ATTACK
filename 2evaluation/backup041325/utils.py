
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

import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import DataLoader, Subset, RandomSampler
torch.set_default_dtype(torch.float32)
import tensorflow_hub as hub
from collections import defaultdict
# torch.autograd.set_detect_anomaly(True)
# from llmrequest import requestNoDf
from names_dataset import NameDataset
nd = NameDataset()

from transformers import logging

# Set the logging level to 'error' to suppress warnings
logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getUSEcosSimilarity(srcDocs, copyDocs):
    # input 2 lists of documents 
    import tensorflow_hub as hub
    embed = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")
    USEcosinSimilarity = []
    sim_metric = torch.nn.CosineSimilarity(dim=1)
    for src, copy in zip(srcDocs, copyDocs):
        emb1, emb2 = embed([src, copy])["outputs"]
        emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
        srcEmb = torch.unsqueeze(emb1, dim=0).to(device) # [embSz] -> [1, embSz]
        advEmb = torch.unsqueeze(emb2, dim=0).to(device)
        es = sim_metric(srcEmb, advEmb)
        USEcosinSimilarity.append(es.item())
    return USEcosinSimilarity

def labelProcessing(data, labels, tgtModel_name):
    if data == 'yelp':
        labels = [1 if l=='2' else 0 for l in labels]
    if data == 'yelp_val':
        labels = [1 if l=='1' else 0 for l in labels]
    if data == 'imdb':
        labels = [1 if l=='1' else 0 for l in labels]
    if data == 'imdb_val':
        labels = [1 if l=='1' else 0 for l in labels]
        
    # labels = [label.strip() for label in labels]
    valid_indices = [inx for inx in range(len(labels))]
    
    if data.startswith('mnli') or data.startswith('snli'):        
        valid_labels = {'contradiction', 'entailment', 'neutral'}
        valid_indices = [index for index, label in enumerate(labels) if label in valid_labels]
        labels = [labels[inx] for inx in valid_indices]

        if tgtModel_name == 'ROBERTA_SNLI':
            print("ROBERTA_SNLI")
            mapping = {'contradiction': 2, 'entailment': 0, 'neutral': 1}
        
        elif tgtModel_name == 'ROBERTA_MNLI':
            print("ROBERTA_MNLI")
            mapping = {'contradiction': 0, 'entailment': 2, 'neutral': 1}
            
        else:
            print("BERT_UNCASED")
            mapping = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        labels = [mapping[e] for e in labels]
    return labels, valid_indices

# Decode a batch of indexes to texts
def decode_batch(input_ids_batch, tokenizer):
    decoded_texts = []
    for input_ids in input_ids_batch:
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    return decoded_texts

# Decode a batch of indexes to texts
def decode_batch_nli(input_ids_batch, tokenizer):
    # print('input_ids_batch', input_ids_batch)
    decoded_tuples = []
    for input_ids in input_ids_batch:
        # print('input_ids', input_ids)
        # decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print('decoded_text', decoded_text)
        text_parts = decoded_text.split("[SEP]")
        print('text_parts', text_parts)
        text_tuple = (text_parts[0], text_parts[1])
        decoded_tuples.append(text_tuple)
    return decoded_tuples

# Decode the logits into text
def decode_logits(logits, tokenizer):
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_texts = [tokenizer.decode(predicted_id, skip_special_tokens=True) for predicted_id in predicted_ids]
    return predicted_texts

def query(genDocs, tgtModel):
    if tgtModel.startswith('SBERT'):
        genDocs = [[sample[0], sample[1]] for sample in genDocs]

    # tgtModel is the link of the target model
    tokenizer = AutoTokenizer.from_pretrained(tgtModel, max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(tgtModel).to(device)
    tokenized_inputs = tokenizer(genDocs, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():  # No need to compute gradients for prediction
        outputs = model(**tokenized_inputs)

    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions, torch.softmax(outputs.logits, dim=1)

def query_ner(doc, tgt_model):
    # doc [bs]
    model = AutoModelForTokenClassification.from_pretrained(tgt_model)

    with torch.no_grad():
            outputs = model(**doc) # [bs, maxDoclen]
    return outputs

def getImportantTokens(genDocsTokenized, tgtModel, searcbOn, fileName, saveInterval=0):
    # genDocs [bS, docRealLen] is list[list[str]]
    importantTokens = []
    for i in range(len(genDocsTokenized)):
        print(i," th doc")
        tokensRanking = {}
        tokDoc = genDocsTokenized[i]
        genDocs = ' '.join(tokDoc)
        _, srcPred = query(genDocs, tgtModel)
        maxInx = torch.argmax(srcPred[0], dim=-1)
        srcProb = srcPred[0][maxInx]# [1,2] -> [2] -> [1]
        
        for j in tqdm(range(len(tokDoc))):
            # if searcbOn == 'premises':
            #     if tokDoc[j] == '[SEP]':
            #         break
            # elif searcbOn == 'hypothesis':
            #     # there must be hypothesis
            #     if tokDoc[j-1] != '[SEP]':
            #         continue
            # else:
            #     print("error target to search on")
            #     break
            srcToken, tokDoc[j] = tokDoc[j], '[MASK]' # can change to <mask> later
            _, pred = query([' '.join(tokDoc)], tgtModel) 
            predInx = torch.argmax(pred[0], dim=-1)
            predProb = pred[0][predInx]
            # predProb = pred[0][0]
            # tokensRanking[j] = abs(srcProb*10000-prob*10000)
            tokensRanking[j] = (srcProb - predProb).item()
            tokDoc[j] = srcToken
        importantTokens.append(tokensRanking)
        
        if saveInterval > 0:
            if (i-1) % saveInterval == 0:
                with open(fileName, 'w') as f:
                    json.dump(importantTokens, f)
                print("save ", fileName, "at sample ", i)
    return importantTokens

# def rankDicts(importanceDicts, numsAttackedTokens, maxCandidatesEachToken=10):
#     totalImportance = sum(importanceDicts[0].values())
#     threshold = totalImportance * numsAttackedTokens # perc
#     sortedTokensImportance = sorted(importanceDicts[0].items(), key=lambda item: item[1], reverse=True)
#     cumulative_importance, numsSelectedTokens = 0, 0
#     for token, importance in sortedTokensImportance:
#         cumulative_importance += importance
#         numsSelectedTokens += 1
#         if (cumulative_importance >= threshold) or (numsSelectedTokens>=maxCandidatesEachToken):
#             break

#     topk_dicts = []
#     for d in importanceDicts:
#         # Sort the dictionary by value in descending order and keep only top k keys
#         topk_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:numsSelectedTokens])
#         topk_dicts.append(topk_dict)
#     return topk_dicts

def rankDicts(dicts, numsMaxCandidates):
    topk_dicts = []
    for d in dicts:
        # d = {key: 1000 for key, value in d.items() if value > 0}
        # Sort the dictionary by value in descending order and keep only top k keys
        
        topk_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:numsMaxCandidates])
        
        # `topk_dict = {key: 1000 for key, value in topk_dict.items()}
        topk_dicts.append(topk_dict)
    return topk_dicts

def rank_dict_ner(dicts):
    topk_dicts = []
    for d in dicts:
        # d = {key: 1000 for key, value in d.items() if value > 0}
        # Sort the dictionary by value in descending order and keep only top k keys
        topk_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
        topk_dict = {k:v for k, v in topk_dict.items() if int(v) > 0}
        # `topk_dict = {key: 1000 for key, value in topk_dict.items()}
        topk_dicts.append(topk_dict)
    return topk_dicts


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()
    
def read_corpus(path, tokenizer, clean=True, MR=True, encoding='utf8', shuffle=False,\
    lower=True, dataName='none', dataPath='none', tgtModel_name='None'):
        if dataName == 'mnli_matched_val' or dataName == 'mnli_mismatched_val' or dataName == 'snli_val' or\
            dataName == 'mnli_matched_test' or dataName == 'mnli_mismatched_test' or dataName == 'snli_test':
            with open(dataPath, 'r', encoding=encoding) as file:
                premises, hypothesis, labels, texts = [], [], [], []
                for i, line in enumerate(file):
                    # if i >= 10000:
                    #     break
                    l, p, h = line.strip().split('\t')
                    labels.append(l)
                    premises.append(p)
                    hypothesis.append(h)
                    texts.append((p,h))
            labels, valid_indices = labelProcessing(data=dataName, labels=labels, tgtModel_name=tgtModel_name)
            texts = [texts[inx] for inx in valid_indices]
            return texts, labels
                    
        elif dataName.startswith('mnli') or dataName.startswith('snli'):
            # Open the JSON Lines file and read the first three entries
            with open(dataPath, 'r', encoding=encoding) as file:
                premises, hypothesis, labels, texts = [], [], [], []
                for i, line in enumerate(file):
                    if i >= 10000:
                        break
                    json_obj = json.loads(line)
                    premises.append(json_obj['sentence1'])
                    hypothesis.append(json_obj['sentence2'])
                    labels.append(json_obj['gold_label']) 
                    #premise_tok = tokenizer.tokenize(json_obj['sentence1'])
                    #hypo_tok = tokenizer.tokenize(json_obj['sentence2'])
                    # text = _truncate_seq_pair(premise_tok, hypo_tok)
                    #texts.append(text)
                    texts.append((json_obj['sentence1'], json_obj['sentence2']))
            labels, valid_indices = labelProcessing(data=dataName, labels=labels, tgtModel_name=tgtModel_name)
            texts = [texts[inx] for inx in valid_indices]
            return texts, labels
        
        texts, labels = [], []
        with open(dataPath, encoding=encoding) as fin:
            for line in fin:
                if dataName == 'yelp':
                    text, label = line.rsplit(',', 1)
                if dataName == 'yelp_val':
                    label, sep, text = line.partition(' ')
                if dataName == 'imdb':
                    label, sep, text = line.partition(' ')
                if dataName == 'imdb_val':
                    label, sep, text = line.partition(' ')
                # if dataName == 'mnlimatched_val':
                #     label, sep, text = line.partition('\t')
                text = clean_str(text.strip()) if clean else text.strip()
                texts.append(text)
                labels.append(label)

        labels, valid_indices = labelProcessing(data=dataName, labels=labels)
        return texts, labels

def _truncate_seq_pair(tokens_a, tokens_b, tokenizer, max_length=512):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    max_length = max_length-3
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    text = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b
    text_ids = tokenizer.convert_tokens_to_ids(text)
    text_from_ids = tokenizer.decode(text_ids)
    return text_from_ids

def tokenize_function(examples, dataName, tokenizer, max_length=512):
    if dataName == 'mnlimatched':
        texts = []
        premises = [e[0] for e in examples]
        hypothesis = [e[1] for e in examples]
        for i in range(len(examples)):
            premis_tok = tokenizer.tokenize(premises[i])
            hypo_tok = tokenizer.tokenize(hypothesis[i])
            text = _truncate_seq_pair(premis_tok, hypo_tok, tokenizer, max_length)
            texts.append(text)
        return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)
    return tokenizer(examples, padding='max_length', truncation=True, max_length=max_length)

class CombLossCompute:
    def __init__(self, ceQuality, ceAdv, tgtModel, alpha):
        self.ceQuality = ceQuality
        self.ceAdv = ceAdv
        self.tgtModel = tgtModel
        self.alpha = alpha

    def getUSEcosSimilarity(self, srcDocs, copyDocs):
    # input 2 lists of documents 
        # import tensorflow_hub as hub
        embed = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")
        USEcosinSimilarity = []
        sim_metric = torch.nn.CosineSimilarity(dim=1)
        for src, copy in zip(srcDocs, copyDocs):
            emb1, emb2 = embed([src, copy])["outputs"]
            emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
            srcEmb = torch.unsqueeze(emb1, dim=0).to(device) # [embSz] -> [1, embSz]
            advEmb = torch.unsqueeze(emb2, dim=0).to(device)
            es = sim_metric(srcEmb, advEmb)
            USEcosinSimilarity.append(es.item())
        return USEcosinSimilarity
        
    def QualityLoss(self, modelOut, srcText, copyText, maxInx, docLen, important_tokens):
        modelOut = modelOut.to(device)
        maxInx = maxInx.to(device)
        
        # response, rewardsTensor, avgLLM = requestNoDf(copyText)
        # if not response:
        #     return None, None, False
        # rewardsTensor = rewardsTensor.to(device)
        
        rewardsTensor = self.getUSEcosSimilarity(srcText, copyText)
        qualRewards = torch.tensor(rewardsTensor).to(device)
        
        # qualRewards = (rewardsTensor/10).repeat_interleave(docLen) # [batch_size]
        qualLoss = self.ceQuality(modelOut.contiguous().view(-1, modelOut.size(-1)), maxInx.contiguous().view(-1))
        qualLoss = qualLoss.view(maxInx.shape[0], docLen)
        
        importance_reward_mask = makeImportanceMask(important_tokens=important_tokens,\
            rows=maxInx.shape[0], cols=docLen, rewards=qualRewards)
        qualLoss = (qualLoss * importance_reward_mask)
        qualLoss = qualLoss.sum()/importance_reward_mask.sum()
        return qualLoss, qualRewards.mean(), True

    def AdvLoss(self, modelOut, srcText, copyText, maxInx, docLen, important_tokens):
        srcpredBinary, srcprob = query(srcText, self.tgtModel)
        predBinary, prob       = query(copyText, self.tgtModel) 
        
        advRewards = []
        for i in range(len(srcprob)):
            if srcpredBinary[i] == 1:
                advRewards.append(prob[i][0]*10.0)
            else:
                advRewards.append(prob[i][1]*10.0)
        advRewards = torch.tensor(advRewards).to(device)
        
        importance_reward_mask = makeImportanceMask(important_tokens=important_tokens,\
            rows=maxInx.shape[0], cols=docLen, rewards=advRewards)
        advLoss = self.ceAdv(modelOut.contiguous().view(-1, modelOut.size(-1)), maxInx.contiguous().view(-1))
        advLoss = advLoss.view(maxInx.shape[0], docLen)
        advLoss = (advLoss * importance_reward_mask)
        advLoss = advLoss.sum() / importance_reward_mask.sum()
        return advLoss, advRewards.mean()
    
    def __call__(self, modelOut, srcText, copyText, maxInx, docLen, important_tokens):        
        qualityLoss, avgLLM, sucFlag = self.QualityLoss(modelOut=modelOut, srcText = srcText, \
            copyText=copyText, maxInx=maxInx, docLen=docLen, important_tokens=important_tokens)
        if not sucFlag:
            return -1.0, -1.0, -1.0, -1.0, -1.0, sucFlag
        AdvLoss, advRewards = self.AdvLoss(modelOut=modelOut, srcText=srcText, copyText=copyText,\
            maxInx = maxInx, docLen=docLen, important_tokens=important_tokens)
        TotalLoss = self.alpha * AdvLoss + (1-self.alpha) * qualityLoss
        return TotalLoss, AdvLoss, qualityLoss, avgLLM, advRewards, sucFlag

class CombLossCompute_nli:
    def __init__(self, ceQuality, ceAdv, tgtModel, alpha):
        self.ceQuality = ceQuality
        self.ceAdv = ceAdv
        self.tgtModel = tgtModel
        self.alpha = alpha

    def getUSEcosSimilarity(self, srcDocs, copyDocs):
    # input 2 lists of documents 
        # import tensorflow_hub as hub
        embed = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")
        USEcosinSimilarity = []
        sim_metric = torch.nn.CosineSimilarity(dim=1)
        for src, copy in zip(srcDocs, copyDocs):
            emb1, emb2 = embed([src, copy])["outputs"]
            emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
            srcEmb = torch.unsqueeze(emb1, dim=0).to(device) # [embSz] -> [1, embSz]
            advEmb = torch.unsqueeze(emb2, dim=0).to(device)
            es = sim_metric(srcEmb, advEmb)
            USEcosinSimilarity.append(es.item())
        return USEcosinSimilarity
        
    def QualityLoss(self, modelOut, srcText, copyText, maxInx, docLen, important_tokens):
        modelOut = modelOut.to(device)
        maxInx = maxInx.to(device)
        
        # response, rewardsTensor, avgLLM = requestNoDf(copyText)
        # if not response:
        #     return None, None, False
        # rewardsTensor = rewardsTensor.to(device)
        
        rewardsTensor = self.getUSEcosSimilarity(srcText, copyText)
        qualRewards = torch.tensor(rewardsTensor).to(device)
        
        # qualRewards = (rewardsTensor/10).repeat_interleave(docLen) # [batch_size]
        qualLoss = self.ceQuality(modelOut.contiguous().view(-1, modelOut.size(-1)), maxInx.contiguous().view(-1))
        qualLoss = qualLoss.view(maxInx.shape[0], docLen)
        
        importance_reward_mask = makeImportanceMask(important_tokens=important_tokens,\
            rows=maxInx.shape[0], cols=docLen, rewards=qualRewards)
        qualLoss = (qualLoss * importance_reward_mask)
        qualLoss = qualLoss.sum()/importance_reward_mask.sum()
        return qualLoss, qualRewards.mean(), True

    def AdvLoss(self, modelOut, src_tuple, copy_tuple, maxInx, docLen, important_tokens):
        srcpredBinary, srcprob = query(src_tuple, self.tgtModel)
        predBinary, prob       = query(copy_tuple, self.tgtModel) 
        
        advRewards = []
        for i in range(len(srcprob)):
            if srcpredBinary[i] == 1:
                advRewards.append(prob[i][0]*10.0)
            else:
                advRewards.append(prob[i][1]*10.0)
        advRewards = torch.tensor(advRewards).to(device)
        
        importance_reward_mask = makeImportanceMask(important_tokens=important_tokens,\
            rows=maxInx.shape[0], cols=docLen, rewards=advRewards)
        advLoss = self.ceAdv(modelOut.contiguous().view(-1, modelOut.size(-1)), maxInx.contiguous().view(-1))
        advLoss = advLoss.view(maxInx.shape[0], docLen)
        advLoss = (advLoss * importance_reward_mask)
        advLoss = advLoss.sum() / importance_reward_mask.sum()
        return advLoss, advRewards.mean()
    
    def __call__(self, modelOut, srcText, copyText, maxInx, docLen, important_tokens, src_tuple, copy_tuple):        
        qualityLoss, avgLLM, sucFlag = self.QualityLoss(modelOut=modelOut, srcText = srcText, \
            copyText=copyText, maxInx=maxInx, docLen=docLen, important_tokens=important_tokens)
        if not sucFlag:
            return -1.0, -1.0, -1.0, -1.0, -1.0, sucFlag
        AdvLoss, advRewards = self.AdvLoss(modelOut=modelOut, src_tuple=src_tuple, copy_tuple=copy_tuple,\
            maxInx=maxInx, docLen=docLen, important_tokens=important_tokens)
        TotalLoss = self.alpha * AdvLoss + (1-self.alpha) * qualityLoss
        return TotalLoss, AdvLoss, qualityLoss, avgLLM, advRewards, sucFlag

def makeImportanceMask(important_tokens, rows, cols, rewards):
    importance_reward_mask = torch.zeros((rows, cols)).to(device)
    for inx, impTokens in enumerate(important_tokens):
        reward = rewards[inx]
        for tokenIds in impTokens.keys():
            importance_reward_mask[inx][int(tokenIds)] = reward
    return importance_reward_mask

def make_loss_mask_ner(atk_inx, rows, cols, rewards):
    """
    Args: atk_inx: [bs, max_doc_len] , index starts from the 1st token
    """
    loss_mask = torch.zeros((rows, cols)).to(device)
    for r_inx, tokens in enumerate(atk_inx):
        for token_inx in tokens:
            loss_mask[r_inx][token_inx] = rewards[r_inx][token_inx]
    return loss_mask
   
class CombLossCompute_ner:
    def __init__(self, cross_entropy, tgtModel, alpha):
        self.cross_entropy = cross_entropy

    # ef name_check(self, s):
    def adv_loss(self, atker_output, src_inx, copy_inx, atk_inx, per_inx):
        """
        Calculate the adversarial loss with weighted cross-entropy.
        Args:
            atker_output: Tensor of shape [bs, max_doc_len, vocab_size], model output logits.
            
            tgt_model: target model for calculating probabilities (NER model).
            important_tokens: List of important tokens for importance masking.
        Returns:
            advLoss: Weighted cross-entropy loss.
            advRewards: Average adversarial reward.
        """

        adv_loss = self.cross_entropy(atker_output.contiguous().view(-1, atker_output.size(-1)), max_inx.contiguous().view(-1))
        adv_loss = adv_loss * masked_rewards
        adv_loss = adv_loss.view(max_inx.size(0), -1)  # [bs, max_doc_len]
        normalized_loss = (adv_loss * masked_rewards).sum() / torch.count_nonzero(masked_rewards)  # Normalize
        return normalized_loss, adv_rewards.mean()
        
    def __call__(self, atker_output, max_inx, src_doc, copy_doc, atk_inx, per_inx):        
        adv_loss, adv_rewards = self.adv_loss(atker_output, max_inx, src_doc, copy_doc, atk_inx, per_inx)
        TotalLoss = self.alpha * AdvLoss + (1-self.alpha) * qualityLoss
        return adv_loss, adv_rewards

def makeHighlightText(DocInxCut, attackedInx, tokenizer, attacker_name='bert-base-cased'):
    if not attackedInx:
        hlTextList = [tokenizer.decode(inx, skip_special_tokens=True) for inx in DocInxCut]
        return hlTextList
    
    hlTextList = []
    for inx in range(len(DocInxCut)):
        hlText = []
        for i in range(len(DocInxCut[inx])):
            Id = DocInxCut[inx][i]
            if str(i) in attackedInx[inx]:
                if attacker_name == 'BERT_distill' or attacker_name == 'google-bert/bert-large-uncased':
                    highlighted_id = [1031, 1031, Id, 1033, 1033]
                else:
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
    
def smoothImportantce(normalized_dict, total_numbers_of_candidates, alpha):
    dict_size = len(normalized_dict)
    total_imporrance = sum(normalized_dict.values())
    smoothed_dict = {k: int((v + alpha) * total_numbers_of_candidates / (total_imporrance + alpha * dict_size)) for k, v in normalized_dict.items()}
    return smoothed_dict

def sample_decode_ner(attacker, tokenizer, tokens, attackedIndex, numCandidatesEachToken, unwanted_words, alpha=0.1):
    candidates = {}
    attackedIndex = attackedIndex[0]
    total_importance = sum(attackedIndex.values())
    num_candidates = {key: (tokenizer.vocab_size//2) for key, value in attackedIndex.items()}
    # num_candidates = {key: (value / total_importance) for key, value in attackedIndex.items()}
    # num_candidates = smoothImportantce(num_candidates, numCandidatesEachToken, alpha)
    
    # with torch.no_grad():
    #     # logits = attacker(**tokens)[:, 1:-1]
    #     logits = attacker(**tokens).logits[:, 1:-1]
    with torch.no_grad():
        outputs = attacker(**tokens)
        # Check if 'logits' exists in the model output
        if hasattr(outputs, "logits"):
            logits = outputs.logits[:, 1:-1]  # Use `.logits` if it exists
        else:
            logits = outputs[:, 1:-1]  # Use direct indexing if `.logits` does not exist
    out = logits.squeeze(dim=0) # [maxDocLen, vocabSize]
    maxInx = torch.argmax(logits, dim=-1)  # tensor int64 [bS=1?, maxDocLen]
    unwanted_words_set = set(unwanted_words)
    
    for i in range(out.shape[0]):
        if str(i) in attackedIndex:
            out_masked = out[i].squeeze(dim=0) # [vocabSize]
            for idx in unwanted_words_set:
            # for idx in unwanted_words:
                out_masked[idx] = float('-inf')
            # candidates[i] = torch.topk(out_masked, num_candidates[str(i)], dim=-1).indices.cpu().tolist()
            candidates[i+1] = torch.topk(out_masked, num_candidates[str(i)], dim=-1).indices.cpu().tolist()
    return maxInx, candidates if candidates else None
    

def sample_decode(attacker, attackerName, input_ids, attention_mask, attackedIndex, numCandidatesEachToken, unwanted_words, alpha=0.1):
    # attackedIndex is a dict
    # numCandidatesEachToken is total disered attacked candidates
    candidates = {}
    attackedIndex = attackedIndex[0]
    nums_tgt_tokens = len(attackedIndex)
    num_candidates = {key: (numCandidatesEachToken//nums_tgt_tokens)+1 for key, value in attackedIndex.items()}
    # total_importance = sum(attackedIndex.values())
    # num_candidates = {key: (value / total_importance) for key, value in attackedIndex.items()}
    # num_candidates = smoothImportantce(num_candidates, numCandidatesEachToken, alpha)
    
    
    if (attackerName == 'BERTMaskedLM') or (attackerName == 'BERT_distill') or (attackerName == 'BERT_nonlinear'):
        logits = attacker(input_ids, attention_mask).logits # [bS=1?, maxDocLen, vocabSize]
    elif (attackerName == 'BERTFineTuned') or (attackerName == 'BERTMaskedLMRandom'):
        logits = attacker(input_ids, attention_mask) # [bS=1?, maxDocLen, vocabSize]
    else:
        print("attackerName", attackerName)
        print("Wrong Attacker Name")

    out = logits.squeeze(dim=0) # [maxDocLen, vocabSize]
    maxInx = torch.argmax(logits, dim=-1)  # tensor int64 [bS=1?, maxDocLen]

    out = out[:, 1:-1]
    unwanted_words_set = set(unwanted_words)
    for i in range(out.shape[0]):
        if str(i) in attackedIndex:
            out_masked = out[i].squeeze(dim=0) # [vocabSize]
            for idx in unwanted_words_set:
            # for idx in unwanted_words:
                out_masked[idx] = float('-inf')

            # candidates[i] = torch.topk(logits, numCandidatesEachToken, dim=-1).indices.cpu().tolist()
            # candidates[i] = torch.topk(out_masked, numCandidatesEachToken, dim=-1).indices.cpu().tolist()
            candidates[i] = torch.topk(out_masked, num_candidates[str(i)], dim=-1).indices.cpu().tolist()
    return maxInx, candidates if candidates else None


# def docReplace(doc, oriProb, oriLabel, attentionIdx, candidates, printCandidates=False, tokenizer=None, tgtModel=None):
#     # doc [bS=1, maxLen-2], a tokenized docuemnt ids, input_ids CUTed
#     # attentionIdx [bS=1, numsAttackedTokens]
#     # candidates [bS=1, attackedTokenNumbers, numCandidatesEachToken] # int64    
#     sourceDoc = deepcopy(doc[0]) # [1, maxDocLen] -> [maxDocLen]
#     attentionIdx = attentionIdx[0]
#     attPattern = {}
#     probData = oriProb[0] # [1,2] -> [2]
#     bestPred = torch.tensor(probData[oriLabel]).to(device)
#     oriLabel = torch.tensor(oriLabel[0]).to(device)
#     cntQueries = 0

#     predBinary = torch.argmax(probData).to(device)
#     if predBinary != oriLabel:
#         attackFlag = True
#         return sourceDoc, cntQueries, attackFlag, probData, {}
    
#     for i in attentionIdx: # for important index in all indexes, identify a position
#         i = int(i)
#         attackFlag = False

#         bestToken = tokenizer.convert_ids_to_tokens(sourceDoc[i]) # best token initualized as the current ground true token
#         bestInx = sourceDoc[i]
#         for chara in tqdm(candidates[i]): # chara is an id, for an index in the candidates of it
#             cntQueries += 1
#             sourceDoc[i] = chara # replace the differnt index as on of the index of its candidates
#             toToken = tokenizer.convert_ids_to_tokens(chara) # record whats token is changed to
#             docText = tokenizer.decode(sourceDoc, skip_special_tokens=True) # decode a new text
#             pred_binary, probData = query(docText, tgtModel) # make prediction on this new text
#             pred_binary, probData = torch.tensor(pred_binary[0]).to(device), torch.tensor(probData[0]).to(device) # [1] -> 1 # [1, 2] -> [2] # record the prob of this new text
            
#             if pred_binary != oriLabel: # if current prediction already diff than the grouth truth
#                 attackFlag = True
#                 attPattern[i] = chara # record changing best token (groud truth token -> adversarial token)
#                 print(bestToken, " -> ", toToken)
#                 return sourceDoc, cntQueries, attackFlag, probData, attPattern
#             else:
#                 # if ori label is pos, the pros prediction should as be less as possible
#                 if (oriLabel == 1) and (probData[oriLabel] < bestPred): # predict less positive
#                     if printCandidates:
#                         print("+ pos" ,bestToken, " -> ", toToken, ' ',\
#                             round(bestPred.item(),6) , " -> ", round(probData[oriLabel].item(),6))
#                     bestToken, bestPred = tokenizer.convert_ids_to_tokens(chara), probData[oriLabel] # if so, record new best token and best pred
#                     bestInx = chara
#                     attPattern[i] = chara

#                 elif (oriLabel == 0) and (probData[oriLabel] < bestPred): # predict less negative
#                     if printCandidates:
#                         print("- neg", bestToken, " -> ", toToken, ' ',\
#                             round(bestPred.item(),6) , " -> ", round(probData[oriLabel].item(),6))
#                     bestToken, bestPred = tokenizer.convert_ids_to_tokens(chara), probData[oriLabel]
#                     bestInx = chara
#                     attPattern[i] = chara
#                 else:
#                     sourceDoc[i] = bestInx # change back
#     advInxCut = sourceDoc
#     return advInxCut, cntQueries, attackFlag, probData, {}
# # '''
# # if 1, nowpred-pred >0
# # if 0, nowpred-pred <0



def docReplace(doc, oriProb, oriLabel, attentionIdx, candidates, printCandidates=False, tokenizer=None, tgtModel=None, mode='classification'):
    # doc [bS=1, maxLen-2], a tokenized docuemnt ids, input_ids CUTed
    # attentionIdx [bS=1, numsAttackedTokens]
    # candidates [bS=1, attackedTokenNumbers, numCandidatesEachToken] # int64
    if mode == 'classification':
        mapping = {1: 'pos', 0: 'neg'}
    elif mode == 'nli':
        mapping = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
                
    sourceDoc = deepcopy(doc[0]) # [1, maxDocLen] -> [maxDocLen]
    attentionIdx = attentionIdx[0]
    attPattern = {}
    probData = oriProb[0] # [1,2] -> [2]
    bestPred = torch.tensor(probData[oriLabel]).to(device)
    oriLabel = torch.tensor(oriLabel[0]).to(device)
    cntQueries = 0

    pred_binary = torch.argmax(probData).to(device)
    if pred_binary != oriLabel:
        attackFlag = True
        return sourceDoc, cntQueries, attackFlag, probData, {}
    for i in attentionIdx: # for important index in all indexes, identify a position
        i = int(i)
        attackFlag = False # do not test [SEP] in nli
        if sourceDoc[i] == 102:
            continue

        bestToken = tokenizer.convert_ids_to_tokens(sourceDoc[i]) # best token initualized as the current ground true token
        bestInx = sourceDoc[i]
        for chara in tqdm(candidates[i]): # chara is an id, for an index in the candidates of it
            cntQueries += 1
            sourceDoc[i] = chara # replace the differnt index as on of the index of its candidates
            toToken = tokenizer.convert_ids_to_tokens(chara) # record whats token is changed to
            docText = tokenizer.decode(sourceDoc, skip_special_tokens=True) # decode a new text
            pred_binary, probData = query(docText, tgtModel) # make prediction on this new text
            pred_binary, probData = torch.tensor(pred_binary[0]).to(device), torch.tensor(probData[0]).to(device) # [1] -> 1 # [1, 2] -> [2] # record the prob of this new text
            
            if pred_binary != oriLabel: # if current prediction already diff than the grouth truth
                attackFlag = True
                attPattern[i] = chara # record changing best token (groud truth token -> adversarial token)
                print(bestToken, " -> ", toToken)
                return sourceDoc, cntQueries, attackFlag, probData, attPattern
            else:
                if (probData[oriLabel] < bestPred):
                    print(mapping[oriLabel.item()], " " ,bestToken, " -> ", toToken, ' ',\
                            round(bestPred.item(),6) , " -> ", round(probData[oriLabel].item(),6))
                    bestToken, bestPred = tokenizer.convert_ids_to_tokens(chara), probData[oriLabel] # if so, record new best token and best pred
                    bestInx = chara
                    attPattern[i] = chara
                else:
                    sourceDoc[i] = bestInx # change back
    advInxCut = sourceDoc
    return advInxCut, cntQueries, attackFlag, probData, {}
# '''
# if 1, nowpred-pred >0
# if 0, nowpred-pred <0



def doc_replace_nli(doc, oriProb, oriLabel, attentionIdx, candidates, printCandidates=False, tokenizer=None, tgtModel=None, mode=None):
    # doc [bS=1, maxLen-2], a tokenized docuemnt ids, input_ids CUTed
    # attentionIdx [bS=1, numsAttackedTokens]
    # candidates [bS=1, attackedTokenNumbers, numCandidatesEachToken] # int64
    # mapping = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
                
    sourceDoc = deepcopy(doc[0]) # [1, maxDocLen] -> [maxDocLen]
    attentionIdx = attentionIdx[0]
    attPattern = {}
    probData = oriProb[0] # [1,2] -> [2]
    bestPred = torch.tensor(probData[oriLabel]).to(device)
    oriLabel = torch.tensor(oriLabel[0]).to(device)
    cntQueries = 0

    pred_binary = torch.argmax(probData).to(device)
    if pred_binary != oriLabel:
        attackFlag = True
        return sourceDoc, cntQueries, attackFlag, probData, {}
    for i in attentionIdx: # for important index in all indexes, identify a position
        i = int(i)
        attackFlag = False # do not test [SEP] in nli
        if sourceDoc[i] == 102:
            continue

        bestToken = tokenizer.convert_ids_to_tokens(sourceDoc[i]) # best token initualized as the current ground true token
        bestInx = sourceDoc[i]
        for chara in tqdm(candidates[i]): # chara is an id, for an index in the candidates of it
            toToken = tokenizer.convert_ids_to_tokens(chara) # record whats token is changed to
            if is_chinese_char(toToken) or toToken is None:
                cntQueries += 1
                continue
            cntQueries += 1
            sourceDoc[i] = chara # replace the differnt index as on of the index of its candidates
            
            # docText = tokenizer.decode(sourceDoc, skip_special_tokens=True) # decode a new text

            decoded_premise, decoded_hypothesis = make_premise_hypothesis_from_indices([101]+sourceDoc+[102], tokenizer)         
            input = [(decoded_premise, decoded_hypothesis)]
            pred_binary, probData = query(input, tgtModel) # make prediction on this new text
            pred_binary, probData = torch.tensor(pred_binary[0]).to(device), torch.tensor(probData[0]).to(device) # [1] -> 1 # [1, 2] -> [2] # record the prob of this new text
            
            if pred_binary != oriLabel: # if current prediction already diff than the grouth truth
                attackFlag = True
                attPattern[i] = chara # record changing best token (groud truth token -> adversarial token)
                print(bestToken, " -> ", toToken)
                return sourceDoc, cntQueries, attackFlag, probData, attPattern
            else:
                # if (probData[oriLabel] < bestPred):
                #     print(mapping[oriLabel.item()], " " ,bestToken, " -> ", toToken, ' ',\
                #             round(bestPred.item(),6) , " -> ", round(probData[oriLabel].item(),6))
                    
                if (probData[oriLabel] < bestPred):
                    print(oriLabel.item(), " " ,bestToken, " -> ", toToken, ' ',\
                            round(bestPred.item(),6) , " -> ", round(probData[oriLabel].item(),6))
                    bestToken, bestPred = tokenizer.convert_ids_to_tokens(chara), probData[oriLabel] # if so, record new best token and best pred
                    bestInx = chara
                    attPattern[i] = chara
                else:
                    sourceDoc[i] = bestInx # change back
    advInxCut = sourceDoc
    return advInxCut, cntQueries, attackFlag, probData, {}
# '''
# if 1, nowpred-pred >0
# if 0, nowpred-pred <0
def convert_to_list(obj):
    if isinstance(obj, int):
        return [obj]
    return obj

def doc_replace_ner(sample_num, subtokens, tokens, \
    src_labels, atk_inx, candidates, tokenizer, tgt_model, sourceDocs, advDocs, srcLabel, advLabel, targetInx, atkFlag, attackPatterns, docLen, cnts, data_labels, model_labels, sourceToks, advToks, atkcandidates):
    # data_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    # model_lables = {0: 'O', 1: 'B-MISC', 2: 'I-MISC', 3: 'B-PER', 4: 'I-PER', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-LOC', 8: 'I-LOC'}
    src_labels = [data_labels[la] for la in src_labels]   # 0-index starts from 1st tokens
    
    # take source indexes, their replacement candidates, ground true, and hyperparameters.
    # iteratively replace and test prediction.
    # subtokens   [bs=1, maxLen+2]                                      # starts from None ends with None
    # tokens     [bs=1, maxLen]                                         # tokens starts from the 1st token
    # src_labels [maxLen]                                               # 0-index starts from 1st tokens
    # atk_inx    [bs=1, maxLen]                                    # 0-index starts from 1st tokens
    # candidates [attackedTokenNumbers, numCandidatesEachToken] # int64 # 0-index starts from [CLS]
    # breakpoint()
    
    subtokens = subtokens[1:-1] # 0-index starts from 1st tokens
    # token_to_subtokens = {}
    # for m in range(len(subtokens)):
    #     if subtokens[m] not in token_to_subtokens:
    #         token_to_subtokens[subtokens[m]] = [m]
    #     else:
    #         token_to_subtokens[subtokens[m]].append(m)
    
    tokens_to_subtokens = defaultdict(list)
    for i in range(len(subtokens)):
        tokens_to_subtokens[subtokens[i]].append(i) # starts from the 1st token
      
    subtokens_to_tokens = defaultdict(list)
    for ki in range(len(subtokens)):
        subtokens_to_tokens[ki] = tokens_to_subtokens[subtokens[ki]] # starts from the 1st token
        
    atk_inx = [str(int(di)) for di in atk_inx[0]] # starts from the 1st token 
    
    tokenized_input = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    for i in atk_inx:
        i = int(i)
        cntQueries = 0
        attPattern = []
        
        ori_tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])
        ori_tok = ori_tokens[i+1]
        ori_label = src_labels[i]
        
        l = src_labels[i]
        # l_ = data_labels[l] # true label
    
        # if l_ in ['B-PER', 'I-PER']:
        # if l_ in ['B-PER']:
        outputs = torch.nn.functional.softmax(query_ner(doc=tokenized_input, tgt_model=tgt_model).logits[:, 1:-1], dim=-1) # starts from the 1st token
        pred_labels = torch.argmax(outputs, dim=2).squeeze().tolist() # [bs=1, maxLen, nums_ner]
        pred_labels = convert_to_list(pred_labels)
        pred_labels = [model_labels[lb] for lb in pred_labels] # starts from the 1st token
        pred_l = pred_labels[i]
        pred_prob = outputs.squeeze(0)[i]
        
        # if pred_l not in ['B-PER', 'I-PER']:
        if pred_l not in ['B-PER']:                
            # res.append([i,
            #             #tokenized_input['input_ids'][0].tolist(),
            #             tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True),
            #             #tokenized_input['input_ids'][0].tolist(),
            #             tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True),
            #             cntQueries,
            #             True,
            #             ori_label,
            #             pred_labels, 
            #             #pred_prob.tolist(), 
            #             attPattern])
            sourceDocs.append(' '.join(tokens))
            sourceToks.append(ori_tok)
            advDocs.append(' '.join(tokens))
            advToks.append(ori_tok)
            srcLabel.append(ori_label)
            advLabel.append(pred_l)
            atkcandidates.append([])
            targetInx.append(i)
            atkFlag.append(1)
            attackPatterns.append({})
            docLen.append(len(subtokens))
            cnts.append(cntQueries)
            
        else:
            candidate = []
            src_copy = copy.deepcopy(tokenized_input)# starts from the 1st token
            # src_copy['input_ids'] = src_copy['input_ids'][:,1:-1]
            for c in tqdm(candidates[i+1]): # starts from the [CLS] token
                src_copy['input_ids'][0][i+1] = c # starts from the [CLS] token
                name = [src_copy['input_ids'][0][ic] for ic in subtokens_to_tokens[i+1]] # starts from the [CLS] token
                name = tokenizer.decode(name)
                name_check = nd.search(name)
                if name_check['first_name'] or name_check['last_name']:
                    with torch.no_grad():
                        cntQueries += 1
                        candidate.append(name)
                        outputs = torch.nn.functional.softmax(query_ner(doc=src_copy, tgt_model=tgt_model).logits[:, 1:-1], dim=-1) # starts from the 1st token
                        pred_labels = torch.argmax(outputs, dim=2).squeeze().tolist()
                        pred_labels = [model_labels[lb] for lb in pred_labels]
                        pred_labels = convert_to_list(pred_labels)
                        pred_l = pred_labels[i]
                        pred_prob = outputs.squeeze(0)[i]
        
                        # if pred_l not in ['B-PER', 'I-PER']:
                        if pred_l not in ['B-PER']:
                            # attPattern.append({src_copy['input_ids'][0][i+1].tolist():c})            
                            # res.append([i,
                            #             # tokenized_input['input_ids'][0].tolist(),
                            #             tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True),
                            #             # src_copy['input_ids'][0].tolist(),
                            #             tokenizer.decode(src_copy['input_ids'][0], skip_special_tokens=True),
                            #             cntQueries,
                            #             True,
                            #             ori_label,
                            #             pred_labels, 
                            #             # pred_prob.tolist(), 
                            #             attPattern])       
                            sourceDocs.append(' '.join(tokens))
                            sourceToks.append(ori_tok)
                            advDocs.append(tokenizer.decode(src_copy['input_ids'][0], skip_special_tokens=True))
                            advToks.append(c)
                            srcLabel.append(ori_label)
                            advLabel.append(pred_l)
                            atkcandidates.append(candidate)
                            targetInx.append(i)
                            atkFlag.append(1)
                            attackPatterns.append({tokenized_input['input_ids'][0][i+1].tolist():c})
                            docLen.append(len(subtokens))
                            cnts.append(cntQueries)
                            break
            else:
                # res.append([i,
                #             # tokenized_input['input_ids'][0].tolist(),
                #             tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True),
                #             # src_copy['input_ids'][0].tolist(),
                #             tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True),
                #             cntQueries,
                #             False,
                #             ori_label,
                #             pred_labels, 
                #             # pred_prob.tolist(), 
                #             attPattern])
                sourceDocs.append(' '.join(tokens))
                sourceToks.append(ori_tok)
                advDocs.append(' '.join(tokens))
                advToks.append(ori_tok)
                srcLabel.append(ori_label)
                advLabel.append(ori_label)
                atkcandidates.append(candidate)
                targetInx.append(i)
                atkFlag.append(0)
                attackPatterns.append({})
                docLen.append(len(subtokens))
                cnts.append(cntQueries)
            
    # the final changed, number queries for each tokens, attack_flag, list of list, list of dict
    return sourceDocs, sourceToks, advDocs, advToks, srcLabel, advLabel, atkcandidates, targetInx, atkFlag, attackPatterns, docLen, cnts
                

    # pred_binary = torch.argmax(probData).to(device)
    # if pred_binary != oriLabel:
    #     attackFlag = True
    #     return sourceDoc, cntQueries, attackFlag, probData, {}
    # for i in attentionIdx: # for important index in all indexes, identify a position
    #     i = int(i)
    #     attackFlag = False # do not test [SEP] in nli
    #     if sourceDoc[i] == 102:
    #         continue

    #     bestToken = tokenizer.convert_ids_to_tokens(sourceDoc[i]) # best token initualized as the current ground true token
    #     bestInx = sourceDoc[i]
    #     for chara in tqdm(candidates[i]): # chara is an id, for an index in the candidates of it
    #         toToken = tokenizer.convert_ids_to_tokens(chara) # record whats token is changed to
    #         if is_chinese_char(toToken) or toToken is None:
    #             cntQueries += 1
    #             continue
    #         cntQueries += 1
    #         sourceDoc[i] = chara # replace the differnt index as on of the index of its candidates
            
    #         # docText = tokenizer.decode(sourceDoc, skip_special_tokens=True) # decode a new text

    #         decoded_premise, decoded_hypothesis = make_premise_hypothesis_from_indices([101]+sourceDoc+[102], tokenizer)         
    #         input = [(decoded_premise, decoded_hypothesis)]
    #         pred_binary, probData = query(input, tgtModel) # make prediction on this new text
    #         pred_binary, probData = torch.tensor(pred_binary[0]).to(device), torch.tensor(probData[0]).to(device) # [1] -> 1 # [1, 2] -> [2] # record the prob of this new text
            
    #         if pred_binary != oriLabel: # if current prediction already diff than the grouth truth
    #             attackFlag = True
    #             attPattern[i] = chara # record changing best token (groud truth token -> adversarial token)
    #             print(bestToken, " -> ", toToken)
    #             return sourceDoc, cntQueries, attackFlag, probData, attPattern
    #         else:
    #             # if (probData[oriLabel] < bestPred):
    #             #     print(mapping[oriLabel.item()], " " ,bestToken, " -> ", toToken, ' ',\
    #             #             round(bestPred.item(),6) , " -> ", round(probData[oriLabel].item(),6))
                    
    #             if (probData[oriLabel] < bestPred):
    #                 print(oriLabel.item(), " " ,bestToken, " -> ", toToken, ' ',\
    #                         round(bestPred.item(),6) , " -> ", round(probData[oriLabel].item(),6))
    #                 bestToken, bestPred = tokenizer.convert_ids_to_tokens(chara), probData[oriLabel] # if so, record new best token and best pred
    #                 bestInx = chara
    #                 attPattern[i] = chara
    #             else:
    #                 sourceDoc[i] = bestInx # change back
    # advInxCut = sourceDoc
    # return advInxCut, cntQueries, attackFlag, probData, {}


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
# from llmrequest import requestNoDf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class Generator(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(d_model, vocab)

#     def forward(self, x):
#         return self.proj(x)

# class BertAttacker(nn.Module):
#     def __init__(self, bertEncoder, generator):
#         super(BertAttacker, self).__init__()
#         self.bert = bertEncoder
#         self.generator = generator
    
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         #hidden_state = outputs.last_hidden_state
#         hidden_state = outputs.hidden_states[-1]
#         logits = self.generator(hidden_state) + 1e-3
#         return logits

# # Set the seed for reproducibility
# def set_seed(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# class Generator(nn.Module):
#     def __init__(self, d_model, vocab, seed=42): # 64
#         super(Generator, self).__init__()
#         self.proj = nn.Linear(d_model, vocab)
#         if seed is not None:
#             set_seed(seed)

#     def forward(self, x):
#         return self.proj(x)

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
        # print("Initialize the weights with Gaussian N(0, 1)")

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
    
def make_premise_hypothesis_from_indices(indices, tokenizer, sep_id = 102):
    # Find the indices of the [SEP] tokens ids
    sep_indices = [i for i, id in enumerate(indices) if id == sep_id]
    # Extract the tokenized premise and hypothesis
    tokenized_premise = indices[1:sep_indices[0]]  # From [CLS] to first [SEP]
    tokenized_hypothesis = indices[sep_indices[0] + 1:]  # After first [SEP]
    # Decode the tokens back to strings
    decoded_premise = tokenizer.decode(tokenized_premise, skip_special_tokens=True)
    decoded_hypothesis = tokenizer.decode(tokenized_hypothesis, skip_special_tokens=True)
    
    return decoded_premise, decoded_hypothesis

def is_chinese_char(text):
    if text is None:
        return False
    for char in text:
        if 0x4E00 <= ord(char) <= 0x9FFF:
            return True
    return False

import subprocess
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd
import subprocess
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv,noheader", "--query-gpu=memory.used,memory.free"]).decode('utf-8')
    gpu_df = pd.read_csv(StringIO(gpu_stats), names=['memory.used', 'memory.free'])
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return 'cuda:{}'.format(idx)