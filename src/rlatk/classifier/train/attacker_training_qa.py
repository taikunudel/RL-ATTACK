import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, pipeline
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import get_scheduler
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, Subset, RandomSampler
import sys
sys.path.append("..")
from rlatk.classifier.utils import *
import random

def select_random_tokens(list_of_lists, num_tokens=5):
    return [random.sample(tokens, min(num_tokens, len(tokens))) for tokens in list_of_lists]

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load tgt model and tokenizer
    tgt_model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    tgt_tokenizer = BertTokenizer.from_pretrained(tgt_model_name)
    tgt_model = BertForQuestionAnswering.from_pretrained(tgt_model_name)

    dataPath = '/usa/taikun/rl-attack/3datasets/squad_dataset'
    attackerPath = 'google-bert/bert-large-cased-whole-word-masking-finetuned-squad'


    # 1. Load the model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(attackerPath)
    vocabSize = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocabSize}")
    # model = AutoModelForQuestionAnswering.from_pretrained(attackerPath)
    config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
    bertEncoder = BertForMaskedLM.from_pretrained("bert-base-cased", config=config)
    for param in bertEncoder.parameters():
        param.requires_grad = False
    generator = Generator(d_model=bertEncoder.config.hidden_size, vocab = vocabSize)
    model = BertAttacker(bertEncoder, generator)
    model.to(device)
    
    # Load dataset
    raw_datasets = load_from_disk(dataPath)
    raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
    # train_data = dataset["train"].select(range(1000))
    # val_data = dataset["validation"].select(range(200))
    max_length = 512
    stride = 128

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        non_tgt_tokens = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

            # tgt_tokens = []
            # # Compute non_tgt_tokens
            # for i in range(len(start_positions)):
            #     tks = [k for k in range(start_positions[i])] + [k for k in range(end_positions[i] + 1)]
            #     tgt_tokens.append(tks)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        # inputs["tgt_tokens"] = tgt_tokens  # Now has uniform shape 
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    train_dataset.set_format(
        type=None,  # Prevent automatic conversion
        columns=[col for col in train_dataset.column_names if col != "non_tgt_tokens"]  # Exclude "non_tgt_tokens"
    )

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    validation_dataset.set_format(
            type=None,  # do not convert everything automatically
            columns=list(validation_dataset.column_names)
        )

    # 2. Create Dataloaders
    #    Suppose 'train_dataset' and 'eval_dataset' are already tokenized and ready
    train_batch_size = 8
    eval_batch_size = 8

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Modify DataLoader to use collator
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=data_collator  # Ensure all samples in a batch have the same length
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        collate_fn=data_collator
    )
    # train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    # val_dataloader = DataLoader(validation_dataset, batch_size=eval_batch_size)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 4. (Optional) Define a learning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    loss = nn.CrossEntropyLoss(reduction='none', ignore_index = tokenizer.pad_token_id)
    # 5. Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # --- Training ---
        model.train()  # put model in train mode
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
            tgt_tokens = []
            for i in range(len(start_positions)):
                tks = [k for k in range(start_positions[i])] + [k for k in range(end_positions[i] + 1, len(input_ids[i]))]
                tgt_tokens.append(tks)
            tgt_tokens = select_random_tokens(tgt_tokens)

            logits = model(input_ids, attention_mask)   
            probabilities = F.softmax(logits, dim=-1)

            maxInx = []
            for pi in range(logits.shape[0]): # [bs, maxDocLen, vocabsize]
                maxi = []
                for oi in range(logits.shape[1]):
                    if oi in tgt_tokens:
                        maxi.append(input_ids[pi][oi])
                    else:
                        sampled_index = torch.multinomial(probabilities[oi], num_samples=1).item()
                        maxi.append(sampled_index)
                maxInx.append(maxi)
            adv_texts = tokenizer.decode(maxInx, skip_special_tokens=True)

            #maxInx = torch.tensor(maxInx).to(device)


            # backward pass
            loss.backward()
            
            # update
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                print(f"  step {step} - loss: {loss.item():.4f}")
        
        # # --- Evaluation ---
        # model.eval()  # put model in eval mode
        # eval_loss = 0.0
        # eval_steps = 0
        # for batch in eval_dataloader:
        #     batch = {k: v.to(device) for k, v in batch.items()}
            
        #     with torch.no_grad():
        #         outputs = model(**batch)
            
        #     eval_loss += outputs.loss.item()
        #     eval_steps += 1
        
        # avg_eval_loss = eval_loss / eval_steps
        # print(f"  Evaluation loss: {avg_eval_loss:.4f}")

    print("Training complete!")


main()