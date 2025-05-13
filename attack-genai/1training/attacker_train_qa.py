#!/usr/bin/env python
# coding: utf-8

# ## 1. Prepare Dataloader
# Load model directly
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    BertForMaskedLM,
    BertConfig,
    DataCollatorWithPadding
)
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float32)
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
        # return logits
        return logits


def main(args=None):
    set_seed(42)
    epochs = 100
    nums_sample_per_question = 1

    # importantTokensFile = './tokens_qa_train.json'
    importantTokensFile = '/usa/taikun/07_transencoder/0dataProcessing/tokens_qa_train.json'
    tgt_model = 'deepset/bert-base-cased-squad2'
    # attacker_model = 'bert-base-cased'
    attacker_model = 'distilbert/distilbert-base-cased'
    
    # Load attacked tokens Json file
    with open(importantTokensFile, 'r') as f:
        importantTokens = json.load(f)
    importantTokens = [json.dumps(line) for line in importantTokens]

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
    
    dataset = load_dataset("squad")
    train_dataset = dataset["train"]
    print(train_dataset)

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

    tokenized_train = train_dataset.map(prepare_train_features, batched=True)
    tokenized_train = tokenized_train.add_column("atk_idx", importantTokens)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def custom_collate_fn(features):
        numeric_keys = ["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"]
        numeric_features = []
        text_keys = ["id", "title", "context", "question", "answers", "atk_idx"]
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
        batch_size=16,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # BERT LINEAR
    config = BertConfig.from_pretrained(attacker_model, output_hidden_states=True)

    bertEncoder = BertForMaskedLM.from_pretrained(attacker_model, config=config).eval()
    for param in bertEncoder.parameters():
        param.requires_grad = False
    generator = Generator(d_model=bertEncoder.config.hidden_size, vocab=vocabSize)
    model = BertAttacker(bertEncoder, generator).to(device)

    # # BERT NONLINEAR
    # # Define the same custom classification head used during training
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
    #         # return logits
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
    # model.to(device)


    # OTHERS
    # model = BertForMaskedLM.from_pretrained(attacker_model, config=config).to(device).eval()
    # for name, param in model.named_parameters():
    #     if 'cls' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    tgt_model = AutoModelForQuestionAnswering.from_pretrained(tgt_model).to(device).eval()

    grad_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    print({k: (v.dtype, v.shape) for k, v in grad_params.items()})
    optimizer = torch.optim.Adam(
        grad_params.values(),
        lr=0.01,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    avg_loss, avg_reward = None, None
    avg_time = None
    
    def avg_update(old_v, new_v):
        if old_v is not None:
            return old_v * 0.99 + 0.01 * new_v
        else:
            return new_v
    
    step = 0
    for epoch in range(epochs):
        for batch in train_dataloader:
            breakpoint()
            start_time = time.time()

            step += 1
            input_ids = batch['input_ids'].to(model.bert.device)
            attention_mask = batch['attention_mask'].to(model.bert.device)
            token_type_ids = batch['token_type_ids'].to(model.bert.device)
            start_positions = batch['start_positions'].to(model.bert.device)
            end_positions = batch['end_positions'].to(model.bert.device)
            importantTokens = [json.loads(line) for line in batch["atk_idx"]]
            
            # Print every 100 steps
            if step % 100 == 0:
                # Decode full input (usually question + context)
                decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                print(f"Step {step}")
                print("Full Input:", decoded_text)

                # Extract answer span using start and end positions
                start_idx = start_positions[0].item()
                end_idx = end_positions[0].item()

                # Ensure indices are valid
                if start_idx <= end_idx and end_idx < input_ids.shape[1]:
                    answer = tokenizer.decode(input_ids[0][start_idx:end_idx+1], skip_special_tokens=True)
                else:
                    answer = "[Invalid Answer Span]"

                print("Extracted Answer:", answer)
                print("=" * 50)

            optimizer.zero_grad()

            with torch.no_grad():
                #action_probs = F.softmax(model(input_ids, attention_mask, token_type_ids), dim=-1) # [bS, maxDocLen, vocab_size]
                action_probs = F.softmax(model(input_ids, attention_mask, token_type_ids).logits, dim=-1) # [bS, maxDocLen, vocab_size]
                qa_logits = tgt_model(input_ids, attention_mask, token_type_ids)
                bidx = torch.arange(qa_logits.start_logits.size(0), device=qa_logits.start_logits.device)
                start_prob = F.softmax(qa_logits.start_logits, dim=-1)[bidx, start_positions]
                end_prob = F.softmax(qa_logits.end_logits, dim=-1)[bidx, end_positions]
                mean_prob = (start_prob + end_prob) / 2
            '''`
            for each sample
            for each per token
            get its sampling
            make copy doc
            
            '''
            copy_ids, copy_masks, copy_type_ids = [], [], []
            src_doc, copy_doc = [], []
            batch_indices, seq_indices, vocab_indices = [], [], []
            src_prob, copy_prob_id, copy_start, copy_end = [], [], [], []

            input_ids_arr = input_ids.tolist()
            attention_mask_arr = attention_mask.tolist()
            token_type_ids_arr = token_type_ids.tolist()
            mean_prob_arr = mean_prob.tolist()
            start_positions_arr = start_positions.tolist()
            end_positions_arr = end_positions.tolist()

            for ai in range(len(batch['input_ids'])): # [bs, maxDocLen, vocabsize]
                atk_inx = [int(k) for k, v in sorted(importantTokens[ai].items(), key=lambda x: x[1], reverse=True)][:5]
                
                src_subword_ids = input_ids_arr[ai] # [maxDocLen, vocabsize]
                if step % 100 == 0 and ai == 0:
                    print('atk_inx: ', tokenizer.convert_ids_to_tokens([src_subword_ids[ainx] for ainx in atk_inx]))

                src_attention_mask = attention_mask_arr[ai]
                src_token_type_ids = token_type_ids_arr[ai]
                ai_reward = mean_prob_arr[ai] # [numLabel]
                ai_start, ai_end = start_positions_arr[ai], end_positions_arr[ai]
                copy_subword_ids = deepcopy(src_subword_ids)
                    
                for rnd in range(1, nums_sample_per_question+1):
                    for bi in atk_inx:
                        distribution = action_probs[ai][bi] # [vocabsize]
                        sampled_id = torch.multinomial(distribution, num_samples=1).item()
                        copy_subword_ids[bi] = sampled_id
                        batch_indices.append(ai)
                        seq_indices.append(bi)
                        vocab_indices.append(sampled_id)
                        src_prob.append(ai_reward)
                        copy_prob_id.append(len(copy_ids))

                    src_doc.append(tokenizer.decode(src_subword_ids, skip_special_tokens=True))
                    copy_ids.append(copy_subword_ids)
                    copy_masks.append(src_attention_mask)
                    copy_type_ids.append(src_token_type_ids)
                    copy_doc.append(tokenizer.decode(copy_subword_ids, skip_special_tokens=True))
                    copy_start.append(ai_start)
                    copy_end.append(ai_end)
            
            if len(src_doc) == 0:
                print('skip update as no sample')
                continue
            
            batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=model.bert.device)
            seq_indices = torch.tensor(seq_indices, dtype=torch.long, device=model.bert.device)
            vocab_indices = torch.tensor(vocab_indices, dtype=torch.long, device=model.bert.device)
            src_prob = torch.tensor(src_prob, dtype=torch.float32, device=model.bert.device)
            copy_prob_id = torch.tensor(copy_prob_id, dtype=torch.long, device=model.bert.device)

            copy_ids = torch.tensor(copy_ids, dtype=torch.long, device=model.bert.device)
            copy_masks = torch.tensor(copy_masks, dtype=torch.long, device=model.bert.device)
            copy_type_ids = torch.tensor(copy_type_ids, dtype=torch.long, device=model.bert.device)
            copy_start = torch.tensor(copy_start, dtype=torch.long, device=model.bert.device)
            copy_end = torch.tensor(copy_end, dtype=torch.long, device=model.bert.device)

            # print('src doc : ', src_doc[0])
            # print('copy doc: ', copy_doc[0])
            
            with torch.no_grad():
                copy_qa_logits = tgt_model(copy_ids, copy_masks, copy_type_ids)
                copy_bidx = torch.arange(copy_qa_logits.start_logits.size(0), device=copy_qa_logits.start_logits.device)
                copy_start_prob = F.softmax(copy_qa_logits.start_logits, dim=-1)[copy_bidx, copy_start]
                copy_end_prob = F.softmax(copy_qa_logits.end_logits, dim=-1)[copy_bidx, copy_end]
                copy_mean_prob = (copy_start_prob + copy_end_prob) / 2
            
            adv_rewards = src_prob - copy_mean_prob[copy_prob_id]            
            # logits = model(input_ids, attention_mask, token_type_ids) # [bS, maxDocLen, vocab_size]
            logits = model(input_ids, attention_mask, token_type_ids).logits # [bS, maxDocLen, vocab_size]
            logprobs = torch.log_softmax(logits, dim=-1)
            atk_logprob = logprobs[batch_indices, seq_indices, vocab_indices]
            loss = torch.mean(-1.0 * adv_rewards * atk_logprob)
                   
            loss.backward()
            optimizer.step()

            avg_loss = avg_update(avg_loss, loss.item())
            avg_reward = avg_update(avg_reward, torch.mean(adv_rewards).item())
            end_time = time.time()
            avg_time = avg_update(avg_time, end_time - start_time)
            print(f"Epoch: {epoch} Step:{step} Speed: {1/avg_time:.3f} iter/s Loss:{avg_loss:.6f} Reward:{avg_reward:.3f}")
        model_path = f"/usa/taikun/07_transencoder/1training/qa/attacker_disllbertlinear_squad_{epoch}_{step}_{avg_reward:.4f}.pth"
        torch.save(model.state_dict(), model_path)
     

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision("high")
    main()