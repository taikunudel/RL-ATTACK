#!/usr/bin/env python
# coding: utf-8

# ## 1. Prepare Dataloader
# Load model directly
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import json
from copy import deepcopy
import re
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertForMaskedLM, BertConfig
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    BertForMaskedLM,
    BertConfig,
    DataCollatorWithPadding
)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import DataLoader, Subset, RandomSampler
from datasets import load_from_disk
torch.set_default_dtype(torch.float32)
# torch.autograd.set_detect_anomaly(True)
import os
import sys
# sys.path.append("..")
sys.path.append("/usa/taikun/rl-attack")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from llmrequest import requestNoDf
from utils import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    importantTokensFile = "/usa/taikun/rl-attack/0dataProcessing/tokens_qa_val.json"
    tgt_model_name = 'deepset/bert-base-cased-squad2'
    tgt_model = BertForQuestionAnswering.from_pretrained(tgt_model_name).to(device)
    attacker_model = 'bert-base-cased'
    # attacker_model = 'distilbert/distilbert-base-cased'
    
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
    
    
    random.seed(42)
    dataset = load_from_disk("/usa/taikun/rl-attack/3datasets/squad_dataset")
    # dataset = load_dataset("squad")
    val_dataset = dataset["validation"]
    random_numbers = [random.randint(0, len(val_dataset)-1) for _ in range(1000)]
    val_dataset = val_dataset.select(random_numbers)
    
    # Load attacked tokens Json file
    with open(importantTokensFile, 'r') as f:
        importantTokens = json.load(f)
    importantTokens = [importantTokens[fi] for fi in random_numbers]
    # importantTokens = [json.dumps(line) for line in importantTokens]

    max_length = 384 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["id"] = []
        tokenized_examples["title"] = []
        tokenized_examples["context"] = []
        tokenized_examples["question"] = []
        tokenized_examples["answers"] = []
        for i, offsets in enumerate(offset_mapping):
            # pad, cls, sep are (0, 0)
            # print(offsets)
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # pad, cls, sep are None
            sequence_ids = tokenized_examples.sequence_ids(i)
            # print(sequence_ids)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            tokenized_examples["id"].append(examples["id"][sample_index])
            tokenized_examples["title"].append(examples["title"][sample_index])
            tokenized_examples["context"].append(examples["context"][sample_index])
            tokenized_examples["question"].append(examples["question"][sample_index])
            tokenized_examples["answers"].append(examples["answers"][sample_index])
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                print("warning this should not happen as we use squad")
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                if len(answers["text"]) > 1:
                    print("multi answer", answers)

                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                
                context_start_idx, context_end_idx = token_start_index, token_end_index

                answer_start_idx, answer_end_idx = None, None
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    answer_start_idx, answer_end_idx = cls_index, cls_index
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    answer_start_idx = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    answer_end_idx = token_end_index + 1
                tokenized_examples["start_positions"].append(answer_start_idx)
                tokenized_examples["end_positions"].append(answer_end_idx)

        return tokenized_examples

    tokenized_val = val_dataset.map(prepare_train_features, batched=True)
    # tokenized_val = tokenized_val.add_column("atk_idx", importantTokens)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def custom_collate_fn(features):
        numeric_keys = ["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"]
        numeric_features = []
        # text_keys = ["id", "title", "context", "question", "answers", "atk_idx"]
        text_keys = ["id", "title", "context", "question", "answers"]
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

    val_dataloader = DataLoader(
        tokenized_val,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # # BERT LINEAR
    # config = BertConfig.from_pretrained(attacker_model, output_hidden_states=True)

    # bertEncoder = BertForMaskedLM.from_pretrained(attacker_model, config=config).eval()
    # for param in bertEncoder.parameters():
    #     param.requires_grad = False
    # generator = Generator(d_model=bertEncoder.config.hidden_size, vocab=vocabSize)
    # testModel = BertAttacker(bertEncoder, generator).to(device)

    # attackerFile = "/usa/taikun/rl-attack/1training/qa/attacker_squad_9_55460_0.6309.pth"
    # state_dict = torch.load(attackerFile, map_location=device)
    # testModel.load_state_dict(state_dict, strict=True)
        
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # testModel.to(device)

    # BERT NONLINEAR
    attackerFile = "/usa/taikun/rl-attack/1training/qa/attacker_bertnonlinear_squad_8_49914_0.6450.pth"
    
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
        
    # Load configurationxx
    attacker_path = "bert-base-cased"  # Change to your attacker path if needed
    config = BertConfig.from_pretrained(attacker_path, output_hidden_states=True)

    # Initialize model
    testModel = BertForMaskedLM.from_pretrained(attacker_path, config=config).to(device)

    # Replace the classification head with the custom one
    custom_head = NonLinearHead(config.hidden_size, config.vocab_size)
    testModel.cls = custom_head  # Replace standard head with custom one
    
    # Freeze all layers except the MLM head
    for name, param in testModel.named_parameters():
        if 'cls' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    testModel.to(device)

    def attack(valid_dataloader, attacker, importantTokens):
        numCandidatesEachToken = 60
        src_doc, src_ans, adv_doc, adv_ans, f1s, cnts, atk_patterns, atk_flags, indices = [], [], [], [], [], [], [], [], []
        total, total_from, advCnt = 0.0, 0.0, 0.0
        totalQueries = 0.0
        total_perturbed_tokens_nums = 0
        total_docs_lens = 0
        
        for batch in valid_dataloader: # batch size is 1
            importantTokens_ = [importantTokens[int(total_from)]]
            attackedIndex = rankDicts(importantTokens_, 5)
            # atk_inx = k: v for k, v in sorted(importantTokens[0].items(), key=lambda x: x[1], reverse=True)][:5]
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            answers = batch['answers']
            # Extract the true answer
            for i in range(input_ids.shape[0]):  # Iterate over batch (though batch size = 1 here)
                start_idx = start_positions[i].item()
                end_idx = end_positions[i].item()

                # Extract the tokenized answer from input_ids
                answer_tokens = input_ids[i][start_idx : end_idx + 1]

                # Convert token IDs to string
                true_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

            total_from += 1
            print(total_from," th document.")
            if total_from % 10 == 0:
                print(src_doc)
                print(src_ans)
                print(adv_doc)
                print(adv_ans)
                print(f1s)
                print(cnts)
                print(atk_patterns)
                print(atk_flags)
                
            
            total += 1
        # generate candidates
            maxInx, candidates = sample_decode_qa(attacker=testModel, 
                                               attackerName=attacker_model, input_ids=input_ids,\
                                                  attention_mask=attention_mask, attackedIndex=attackedIndex,\
                                                    numCandidatesEachToken=numCandidatesEachToken,\
                                                          unwanted_words=set())
            if total_from % 10 == 0:
                results = {
                "src_doc": src_doc,
                "src_ans": src_ans,
                "adv_doc": adv_doc,
                "adv_ans": adv_ans,
                "f1s": f1s,
                "cnts": cnts,
                "atk_patterns": atk_patterns,
                "atk_flags": atk_flags,
                "indices": indices}
                # filename = 'squad_bert_random.json'
                filename = 'squad_bert_nonlinear.json'
                # Save the dictionary to a JSON file
                with open(filename, 'w') as json_file:
                    json.dump(results, json_file, indent=4)
                print(f"Results saved to {filename}")

            src_doc, src_ans, adv_doc, adv_ans, f1s, cnts, atk_patterns, atk_flags, indices \
                 = doc_replace_qa(tgt_model, tokenizer, answers, input_ids, attention_mask, token_type_ids, attackedIndex, candidates,\
                    src_doc, src_ans, adv_doc, adv_ans, f1s, cnts, atk_patterns, atk_flags, indices)

    # results = {
    # "src_doc": src_doc,
    # "src_ans": src_ans,
    # "adv_doc": adv_doc,
    # "adv_ans": adv_ans,
    # "f1s": f1s,
    # "cnts": cnts,
    # "atk_patterns": atk_patterns,
    # "atk_flags": atk_flags,
    # "atkFlag": atkFlag,
    # "indices": indices}
    # filename = 'squad_bert_random.json'
    # # Save the dictionary to a JSON file
    # with open(filename, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)
    # print(f"Results saved to {filename}")        
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
                

        # # ruturn adversarial sample and prediction
        # return sourceDocs, sourceLabels, advDocs, srcHighlightedDocs, advHighlightedDocs,\
        #     total, advCnt, avgQueries, attSucFlags, attPatterns, sourcePredProb, advPredProb,\
        #         targetInx, targetTokens, toTokens, docLen


    src_doc, src_ans, adv_doc, adv_ans, f1s, cnts, atk_patterns, atk_flags, indices =\
                        attack(val_dataloader, testModel, importantTokens)

if __name__ == "__main__":
    main()