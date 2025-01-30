import torch
import torch.nn as nn

input_text = "I am a UCI stuent"
input_text_list = input_text.split()

# str2idx
str2idx = {word:idx for idx, word in enumerate(input_text_list)}

# idx2str
idx2str = {idx:str for idx, word in enumerate(input_text_list)}

# input text list to idx
output = [ str2idx[word] for word in input_text_list]

# embed layer
embed_dim = 16
embed_layer = nn.Embedding(len(output), embed_dim)

# input token pass through embed layer
embedded_input = embed_layer(torch.tensor(output))

# unsqueeze
embedded_output = embedded_input.unsqueeze(0)

# positional encoding
max_idx = 12

# define input data
index_ids = torch.arange(len(output), dtype=torch.long)
index_ids = index_ids.unsqueeze(0)

# embed layer
positional_encoding_layer = nn.Embedding(max_idx, embed_dim)

# positional token pass through embed layer
embedded_ids = positional_encoding_layer(index_ids)

# addition
tf_input = embedded_output + embedded_ids

# (embed_dim, head_dim) layer
head_dim = 16
key_layer = nn.Linear(embed_dim, head_dim)
query_layer = nn.Linear(embed_dim, head_dim)
value_layer = nn.Linear(embed_dim, head_dim)

# Key, Query, Value With NN
keys = key_layer(tf_input)
querys = query_layer(tf_input)
values = value_layer(tf_input)

from math import sqrt
import torch.nn.functional as F

def compute_attention(q, k, v, is_casual=False):
    # 1. Q * K.T
    q_k = q @ k.transpose(-2, -1)

    # 2. devide by sqrt of embedding dimension
    dim_k = q.size(-1)
    normalized_q_k = q_k / sqrt(dim_k)

    # masked attention for decoder
    if is_casual:
        q_dim, k_dim = q.size(2), k.size(2)
        mask = torch.ones(q_dim, k_dim, dtype=torch.bool).tril(0)
        normalized_q_k = normalized_q_k.masked_fill(mask == False, float('-inf'))

    # 3. softmax (score)
    score = F.softmax(normalized_q_k, dim=1)

    # 4. score * V
    attention = score @ v 

    return attention

# Multi Head Attention as Class
class MultiheadAttentionHead(nn.Module):
    def __init__(self, token_embed_dim, head_dim, n_head, is_casual=True):
        super().__init__()
        self._n_head = n_head
        self._is_casual = is_casual
        self.key_layer = nn.Linear(embed_dim, head_dim)
        self.query_layer = nn.Linear(embed_dim, head_dim)
        self.value_layer = nn.Linear(embed_dim, head_dim)
        self.concat_layer = nn.Linear(head_dim, head_dim)

    def forward(self, q, k, v, is_casual):
        B, T, C = q.size()
        # Device each Q, K, V into head, then transpose
        mh_q = self.query_layer(q).view(B, T, self._n_head, C//self._n_head).transpose(1, 2)
        mh_k = self.key_layer(k).view(B, T, self._n_head, C//self._n_head).transpose(1, 2)
        mh_v = self.key_layer(v).view(B, T, self._n_head, C//self._n_head).transpose(1, 2)

        # compute attention
        attention = compute_attention(mh_q, mh_k, mh_v, is_casual=is_casual)

        # transpose back, check memory order, concatenate back 
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)

        # Pass through linear layer again
        output = self.concat_layer(attention)

        return output

# Pre-Layer norm + FCNN feedforward class
class PreLayerNormFeedForward(nn.Module):
    def __init__(self, token_embed_dim, ff_layer_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(token_embed_dim)
        self.linear1 = nn.Linear(token_embed_dim, ff_layer_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_layer_dim, token_embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        normed_x = self.norm(x)
        x = normed_x + self.linear2(self.dropout1(self.activation(self.linear1(normed_x))))
        return self.dropout2(x)

# Transformer Encoder - one layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ff_layer_dim, dropout):
        super().__init__()
        # norm layer / attention layer / dropout / ff layer
        self.norm = nn.LayerNorm(d_model)
        self.mh_attention = MultiheadAttentionHead(d_model, d_model, n_head)
        self.dropout = nn.Dropout(dropout)
        self.ff_layer = PreLayerNormFeedForward(d_model, ff_layer_dim, dropout)

    def forward(self, x):
        # norm => attention => dropout => residual => ff
        normed_x = self.norm(x)
        attened_x = normed_x + self.dropout(self.mh_attention(normed_x, normed_x, normed_x, is_casual=False))
        output  = self.ff_layer(attened_x)
        return output 

# Transformer Encoder - multi layer
import copy
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, module, layer_num):
        super().__init__()
        self.clones = get_clones(module, layer_num)

    def forward(self, x):
        output = x
        for clone in self.clones:
            output = clone(output)
        return output

num_head = 4
tf_encoder_layer = TransformerEncoderLayer(embed_dim, num_head, embed_dim*2, 0.5)
tf_layer_output = tf_encoder_layer(tf_input)

num_tf_layer = 5
tf_encoder = TransformerEncoder(tf_encoder_layer, num_tf_layer)
tf_output = tf_encoder(tf_input)

print(f"tf_input.shape: {tf_input.shape}")
print(f"tf_output.shape: {tf_output.shape}")
print(f"tf_layer_output.shape: {tf_layer_output.shape}")