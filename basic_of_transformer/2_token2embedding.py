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

# print output dimension
print(f"Embedded output: {embedded_output}")
print(f"Embedded output.shape: {embedded_output.shape}")

