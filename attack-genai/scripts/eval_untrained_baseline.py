#!/usr/bin/env python
"""Evaluate untrained BERT baseline on the validation set.
Computes the same metrics as the training loop: val_accuracy, loss, total/adv/sem rewards.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from transformers import AutoTokenizer, BertForMaskedLM, BertConfig
from tqdm import tqdm

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_hub as hub

# Reuse functions from the training script
from train_attacker_genai import (
    build_datasets, apply_random_masks, logits_to_labels_doc,
    getUSEcosSimilarity, get_raw_logits
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_untrained(num_doc_masks, alpha, server_url="http://localhost:8002/v1"):
    atker_path = "bert-base-uncased"
    len_doc_max = 512

    USE = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")

    tokenizer = AutoTokenizer.from_pretrained(atker_path, max_length=len_doc_max)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    _, validation_dataloader, _ = build_datasets(
        tokenizer=tokenizer, num_doc_masks=num_doc_masks,
        max_len=len_doc_max, atk_what='doc', seed=42
    )

    config = BertConfig.from_pretrained(atker_path, output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained(atker_path, config=config).to(device)

    # Freeze everything — untrained baseline
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    val_acc_metric = Accuracy(task="binary").to(device)
    all_losses = []
    all_total_rewards = []
    all_adv_rewards = []
    all_sem_rewards = []

    with torch.no_grad():
        bar = tqdm(validation_dataloader, desc=f"Untrained α={alpha} masks={num_doc_masks}")
        for batch in bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            masked_input_ids, mask_positions = apply_random_masks(
                input_ids, attention_mask, num_masks=num_doc_masks, mask_token_id=mask_token_id
            )
            logits = model(masked_input_ids, attention_mask).logits
            batch_size, vocab_size, generated_labels, source_documents, generated_documents = logits_to_labels_doc(
                input_ids, mask_positions, logits, tokenizer, pad_token_id, cls_token_id, sep_token_id
            )

            # Get victim predictions
            _, predictions_, probs_ = get_raw_logits.process_file(data=generated_documents, server_url=server_url)
            predicted_classes = torch.tensor(predictions_).to(device)
            val_acc_metric.update(predicted_classes, labels)

            # Compute rewards (same as training)
            adv_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
            sem_rewards = torch.zeros(batch_size, dtype=torch.float, device=device)
            rewards = torch.zeros(batch_size, dtype=torch.float, device=device)

            USE_res = getUSEcosSimilarity(source_documents, generated_documents, embed=USE)
            for i in range(batch_size):
                if predictions_[i] not in (1, 0):
                    adv_rewards[i] = 0.0 if labels[i] == 0 else 1.0
                elif labels[i] == predictions_[i]:
                    adv_rewards[i] = 1 - probs_[i]
                else:
                    adv_rewards[i] = probs_[i]

                sem_rewards[i] = USE_res[i]
                rewards[i] = alpha * adv_rewards[i] + (1 - alpha) * sem_rewards[i]

            # Compute loss (same as training)
            loss = torch.tensor(0.0, device=device)
            for i in range(batch_size):
                if not mask_positions[i]:
                    continue
                batch_losses = []
                for j, pos in enumerate(mask_positions[i]):
                    token_logits = logits[i, pos]
                    token_label = torch.tensor(generated_labels[i][j], device=device)
                    token_loss = F.cross_entropy(token_logits.unsqueeze(0), token_label.unsqueeze(0))
                    batch_losses.append(token_loss)
                if batch_losses:
                    batch_loss = torch.mean(torch.stack(batch_losses))
                    loss += batch_loss * rewards[i]
            loss /= batch_size

            all_losses.append(loss.item())
            all_total_rewards.append(rewards.mean().item())
            all_adv_rewards.append(adv_rewards.mean().item())
            all_sem_rewards.append(sem_rewards.mean().item())

            bar.set_postfix({
                "acc": f"{val_acc_metric.compute():.4f}",
                "loss": f"{loss.item():.4f}",
                "total": f"{rewards.mean().item():.4f}",
                "adv": f"{adv_rewards.mean().item():.4f}",
                "sem": f"{sem_rewards.mean().item():.4f}",
            })

    val_accuracy = val_acc_metric.compute().item()
    avg_loss = np.mean(all_losses)
    avg_total = np.mean(all_total_rewards)
    avg_adv = np.mean(all_adv_rewards)
    avg_sem = np.mean(all_sem_rewards)

    print(f"\n=== UNTRAINED BASELINE: alpha={alpha}, masks={num_doc_masks} ===")
    print(f"  Val Accuracy: {val_accuracy:.4f}")
    print(f"  Avg Loss:     {avg_loss:.4f}")
    print(f"  Avg Total:    {avg_total:.4f}")
    print(f"  Avg Adv:      {avg_adv:.4f}")
    print(f"  Avg Sem:      {avg_sem:.4f}")

    return val_accuracy, avg_loss, avg_total, avg_adv, avg_sem


if __name__ == "__main__":
    server_url = "http://localhost:8002/v1"
    results = []
    for alpha in [0.3, 0.5, 0.7]:
        for masks in [3, 5, 10]:
            acc, loss, total, adv, sem = evaluate_untrained(masks, alpha, server_url)
            results.append((alpha, masks, acc, loss, total, adv, sem))

    print("\n\n=== SUMMARY ===")
    print(f"{'Alpha':>6} {'Masks':>6} {'Acc':>8} {'Loss':>8} {'Total':>8} {'Adv':>8} {'Sem':>8}")
    print("-" * 62)
    for alpha, masks, acc, loss, total, adv, sem in results:
        print(f"{alpha:>6.1f} {masks:>6d} {acc:>8.4f} {loss:>8.4f} {total:>8.4f} {adv:>8.4f} {sem:>8.4f}")
