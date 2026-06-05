import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding
)
import json
from datasets import load_dataset
import copy
from tqdm import tqdm
from itertools import chain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        padding=False,
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
    tokenized_examples["atk_idx"] = []
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
            attack_idx = list(range(context_start_idx, answer_start_idx)) + list(range(answer_end_idx + 1, context_end_idx + 1))
            tokenized_examples['atk_idx'].append(json.dumps({k: 1 for k in attack_idx}))

    return tokenized_examples

# Define max_length and doc_stride for handling long documents
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

tgt_name = 'deepset/bert-base-cased-squad2'
tokenizer = AutoTokenizer.from_pretrained(tgt_name, max_length=512)
vocabSize = tokenizer.vocab_size
print(f"Vocabulary Size: {vocabSize}")

dataset = load_dataset("squad")
# train_dataset = dataset["train"]
train_dataset = dataset["validation"]
tokenized_train = train_dataset.map(prepare_train_features, batched=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tgt_name = "deepset/bert-base-cased-squad2"
tgt_model = AutoModelForQuestionAnswering.from_pretrained(tgt_name).to(device).eval()

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

importance_files = []
# Iterate through dataset and tokenize one by one
for example in tqdm(tokenized_train):
    input_ids = example["input_ids"]
    atk_inx = json.loads(example["atk_idx"])
    atk_inx = [int(k) for k, v in atk_inx.items() if v > 0]
    start_positions = example['start_positions']
    end_positions = example['end_positions']

    batch_examples = [copy.deepcopy(example)]

    for inx in atk_inx:
        example_copy = copy.deepcopy(example)
        example_copy['input_ids'][inx] = tokenizer.mask_token_id
        batch_examples.append(example_copy)
    
    batch = custom_collate_fn(batch_examples)
    input_ids = batch['input_ids'].to(tgt_model.bert.device)
    attention_mask = batch['attention_mask'].to(tgt_model.bert.device)
    token_type_ids = batch['token_type_ids'].to(tgt_model.bert.device)
    start_positions = batch['start_positions'].to(tgt_model.bert.device)
    end_positions = batch['end_positions'].to(tgt_model.bert.device)

    with torch.inference_mode():
        outputs = tgt_model(input_ids, attention_mask, token_type_ids)
    batch_index = torch.arange(len(batch_examples), dtype=torch.int, device=tgt_model.device)
    start_prob = torch.softmax(outputs['start_logits'], dim=-1)[batch_index, start_positions]
    end_prob = torch.softmax(outputs['end_logits'], dim=-1)[batch_index, end_positions]
    avg_prob = start_prob + end_prob
    impact = (avg_prob[0] - avg_prob[1:]).tolist()
    impacts = {atk_inx[i]: impact[i] for i in range(len(atk_inx))}

    print(tokenizer.decode(example["input_ids"], skip_special_tokens=True))
    print(example['start_positions'], example['end_positions'], tokenizer.decode(example["input_ids"][example['start_positions']:example['end_positions']+1], skip_special_tokens=True))
    for idx, diff in sorted(impacts.items(), key=lambda x:x[1], reverse=True)[:10]:
        print(idx, diff, tokenizer.decode(example["input_ids"][idx]))
    importance_files.append(impacts)

with open("tokens_qa_train.json", "w") as json_file:
    json.dump(importance_files, json_file, indent=4)
    print("Importance files saved to tokens_qa_train.json")