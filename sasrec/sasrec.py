import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm

class SelfAttentionSessionEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_head,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps
    ):
        super(SelfAttentionSessionEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_eps = layer_norm_eps

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_head,
            dropout=attn_dropout_prob,
            batch_first=True
        )
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )

        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_tensor, attention_mask, padding_mask):
        input_tensor = self.LayerNorm(input_tensor)
            
        self_attention_output, _ = self.self_attention(
            input_tensor, input_tensor, input_tensor, attn_mask=attention_mask
        )
        
        self_attention_output = input_tensor + self_attention_output
        self_attention_output = self.LayerNorm(self_attention_output)

        feedforward_output = self.feedforward(self_attention_output)
        feedforward_output = input_tensor + feedforward_output
        feedforward_output = feedforward_output * padding_mask.unsqueeze(-1)
        return feedforward_output
    


class SelfAttentiveSessionEncoder(nn.Module):
    def __init__(
        self,
        num_items,
        hidden_size = 50,
        n_head = 1,
        hidden_dropout_prob = 0.5,
        attn_dropout_prob = 0.5,
        layer_norm_eps = 1e-5,
        n_layers = 2,
        max_session_length = 10,
        logit_type = 'dot',
        scale = 1.0,
        batchnorm = False
    ):
        super(SelfAttentiveSessionEncoder, self).__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.n_layers = n_layers
        self.max_session_length = max_session_length
        self.logit_type = logit_type
        self.scale = scale
        self.batchnorm = batchnorm
        # define layers
        self.item_embedding = nn.Embedding(num_items, hidden_size)
        self.classifier_embedding = nn.Embedding(num_items, hidden_size)

        self.position_embedding = nn.Embedding(self.max_session_length, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layers = nn.ModuleList([
            SelfAttentionSessionEncoderLayer(
                hidden_size,
                n_head,
                hidden_dropout_prob,
                attn_dropout_prob,
                layer_norm_eps
            ) for _ in range(n_layers)
        ])

        if self.batchnorm:
            self.item_bn = nn.BatchNorm1d(self.hidden_size)
            self.sess_bn = nn.BatchNorm1d(self.hidden_size)

        self.iteration = 0

    def get_attention_mask(self,item_seq, bidirectional=False):
        # Create attention mask to achieve future blinding
        if bidirectional:
            attention_mask = torch.zeros(item_seq.size(1), item_seq.size(1), device=item_seq.device).to(torch.bool)
        else:
            attention_mask = torch.triu(torch.ones(item_seq.size(1), item_seq.size(1), device=item_seq.device), diagonal=1).to(torch.bool)
        return attention_mask
    
    def get_padding_mask(self, item_seq):
        # Create padding mask to ignore padding items in the sequence
        padding_mask = (item_seq == 0).bool()
        return padding_mask
    
    def get_item_embedding(self):
        item_embedding = self.item_embedding.weight

        return item_embedding
    
    def item_emb_cos_reg(self):
        item_embedding = self.get_item_embedding()
        normalized_item_embedding = torch.nn.functional.normalize(item_embedding, dim=-1)
        reg_loss = torch.sum(torch.matmul(normalized_item_embedding, normalized_item_embedding.T)) / (item_embedding.size(0) ** 2)
        return reg_loss
    
    def get_session_embedding(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_embedding = self.item_embedding(item_seq)

        input_embedding = item_embedding + position_embedding
        input_embedding = self.dropout(input_embedding) # [batch_size, max_session_length, hidden_size]

        padding_mask_bool = self.get_padding_mask(item_seq)
        padding_mask = torch.where(padding_mask_bool, torch.zeros_like(padding_mask_bool), torch.ones_like(padding_mask_bool))
        attention_mask = self.get_attention_mask(item_seq)

        for layer in self.layers:
            input_embedding = layer(input_embedding, attention_mask, padding_mask)
        
        output_embedding = input_embedding
        # take the embedding of the last item
        session_embedding = output_embedding[:, -1, :] # [batch_size, hidden_size]

        return session_embedding

    
    def forward(self, item_seq):

        session_embedding = self.get_session_embedding(item_seq)
        item_embedding = self.get_item_embedding()

        if self.batchnorm:
            session_embedding = self.sess_bn(session_embedding)
            item_embedding = self.item_bn(item_embedding)

        if self.logit_type == 'dot':
            logits = torch.matmul(session_embedding, item_embedding.T)
        elif self.logit_type == 'euclidean':
            logits = -torch.cdist(session_embedding, item_embedding, p=2)
            logits = logits * self.scale
        else:
            raise ValueError(f"Invalid logit type: {self.logit_type}")

        return logits