input_text = "I am a UCI stuent"
input_text_list = input_text.split()

# str2idx
str2idx = {word:idx for idx, word in enumerate(input_text_list)}

# idx2str
idx2str = {idx:str for idx, word in enumerate(input_text_list)}

# input text list to idx
output = [ str2idx[word] for word in input_text_list]

# print output
print(f"output: {output}")