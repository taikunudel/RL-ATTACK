#!/usr/bin/env python
# coding: utf-8
"""Attacker-encoder factory: build the masked-LM attacker from ANY HF encoder.

This replaces the hard-wired ``BertForMaskedLM`` / ``BertConfig`` construction so
the attacker backbone can be swapped for ablations (``bert-base-uncased``,
``distilbert-base-uncased``, ``roberta-base``, ...) just by changing
``--atker_path``.

Parity
------
For ``bert-base-uncased`` the ``Auto*`` classes resolve to exactly
``BertForMaskedLM`` / ``BertConfig``, and the freeze rule below selects exactly
the same parameter set as the old ``'cls' in name`` rule, so behaviour is
unchanged.

Two heads
---------
* ``linear_head=True``  (default): the model's built-in MLM head -- only the head
  is trained, the encoder trunk is frozen. This is the path every existing
  result was produced with.
* ``linear_head=False``: a 3-layer FFN head (``MaskFillingHead``) on top of the
  encoder's last hidden state. The previous ``BertForJailbreak`` implementation
  of this path never actually ran (it returned a bare tensor while the training
  loop expected ``.logits``, and read a non-existent ``last_hidden_state`` off a
  masked-LM output); this version fixes both so the FFN head works.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput


class MaskFillingHead(nn.Module):
    """3-layer FFN that maps an encoder hidden state to vocab logits."""

    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(MaskFillingHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, vocab_size)  # Output to vocab size for mask filling

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class AttackerWithFFNHead(nn.Module):
    """Encoder backbone + 3-layer FFN mask-filling head.

    Generalises the old ``BertForJailbreak`` to any HF encoder. ``forward``
    returns a ``MaskedLMOutput`` so callers can uniformly read ``.logits``
    (matching the built-in MLM head's interface).
    """

    def __init__(self, atker_path, device, ffn_hidden_dim=256):
        super(AttackerWithFFNHead, self).__init__()
        self.config = AutoConfig.from_pretrained(atker_path)
        self.encoder = AutoModel.from_pretrained(atker_path)
        self.classification_head = MaskFillingHead(
            self.config.hidden_size, ffn_hidden_dim, self.config.vocab_size
        ).to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classification_head(outputs.last_hidden_state)
        return MaskedLMOutput(logits=logits)


def _freeze_trunk_train_head(model):
    """Freeze the encoder trunk, leave only the masked-LM head trainable.

    Uses ``model.base_model`` (the encoder trunk) to identify trunk params. For
    BERT this is exactly the old ``'cls' in name`` rule; for other encoders it
    correctly trains their LM head (``lm_head`` / ``vocab_projector`` / ...).
    """
    trunk_param_ids = {id(p) for p in model.base_model.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) not in trunk_param_ids


def build_attacker(atker_path, linear_head=True, device="cpu"):
    """Construct the attacker model and set up trainable parameters.

    Returns a model whose ``forward(input_ids, attention_mask)`` exposes
    ``.logits`` of shape ``[batch, seq_len, vocab_size]`` for both head types.
    """
    if linear_head:
        print("Using MLM Head (built-in masked-LM head)")
        config = AutoConfig.from_pretrained(atker_path, output_hidden_states=True)
        model = AutoModelForMaskedLM.from_pretrained(atker_path, config=config).to(device)
        _freeze_trunk_train_head(model)
    else:
        print("Using MaskFillingHead (3-layer FFN)")
        model = AttackerWithFFNHead(atker_path, device).to(device)
        for name, param in model.named_parameters():
            param.requires_grad = "classification_head" in name
    return model
