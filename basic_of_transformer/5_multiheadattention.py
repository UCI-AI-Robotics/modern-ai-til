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

# print output
print(f"token: {output}")

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

    def forward(self, q, k, v):
        B, T, C = q.size()
        # Device each Q, K, V into head, then transpose
        mh_q = self.query_layer(q).view(B, T, self._n_head, C//self._n_head).transpose(1, 2)
        mh_k = self.key_layer(k).view(B, T, self._n_head, C//self._n_head).transpose(1, 2)
        mh_v = self.key_layer(v).view(B, T, self._n_head, C//self._n_head).transpose(1, 2)

        # compute attention
        attention = compute_attention(mh_q, mh_k, mh_v)

        # transpose back, check memory order, concatenate back 
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)

        # Pass through linear layer again
        output = self.concat_layer(attention)

        return output

num_head = 4
multi_attention_head = MultiheadAttentionHead(embed_dim, head_dim, 4)
self_attention_from_class = multi_attention_head(querys, keys, values)
print(f"self_attention_from_class.shape: {self_attention_from_class.shape}") 