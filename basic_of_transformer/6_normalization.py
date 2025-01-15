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

# nn.LayerNorm
layer_norm = nn.LayerNorm(embed_dim)

# embedded token through layer norm 
normed_ids = layer_norm(tf_input)

# check mean and std var
mean = normed_ids.mean(dim=-1)
std = normed_ids.std(dim=-1)

print(f"mean/std : {mean}/{std}")