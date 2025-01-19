from datasets import load_dataset, Dataset
import pandas as pd
import random

# import dataset from hg hub
klue_mrc_dataset = load_dataset('klue', 'mrc')
klue_mrc_dataset_train = load_dataset('klue', 'mrc', split="train")

# import dataset from local dirc
local_data = load_dataset("csv", data_files="my_file.csv")

# python dict to dataset
test_dict = {
    'train': {
        'text': ['Sample sentence 1', 'Sample sentence 2', 'Sample sentence 3', 
                 'Sample sentence 4', 'Sample sentence 5', 'Sample sentence 6', 
                 'Sample sentence 7', 'Sample sentence 8', 'Sample sentence 9', 
                 'Sample sentence 10'],
        'label': ['positive', 'positive', 'neutral', 'positive', 'positive', 
                  'negative', 'positive', 'positive', 'positive', 'positive']
    },
    'validation': {
        'text': ['Validation sentence 1', 'Validation sentence 2', 
                 'Validation sentence 3', 'Validation sentence 4', 
                 'Validation sentence 5'],
        'label': ['positive', 'positive', 'neutral', 'positive', 'negative']
    },
    'test': {
        'text': ['Test sentence 1', 'Test sentence 2', 'Test sentence 3', 
                 'Test sentence 4', 'Test sentence 5'],
        'label': ['positive', 'neutral', 'negative', 'neutral', 'neutral']
    }
}
dict_dataset = Dataset.from_dict(test_dict)

# pandas df as dataset
test_pd = pd.DataFrame(test_dict)
pd_dataset = Dataset.from_pandas(test_pd)

print(f"klue_mrc_dataset_train: {klue_mrc_dataset_train}")
print(f"local_data: {local_data}")
print(f"dict_dataset: {dict_dataset}")
print(f"pd_dataset: {pd_dataset}")