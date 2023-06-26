import pandas as pd
import simplejson
import rapidfuzz
from rapidfuzz import process, utils
from tqdm.auto import tqdm
import os
import glob

# Load dataset

newoag=pd.read_json('oag_qa_20230512.json',lines=True)
newoag_title=newoag[['title','answers']]
print(f'newoag_title dataset size: {newoag_title.shape}')

folder_path = r".\oagqa-topic-v2"
files = glob.glob(os.path.join(folder_path, "*train*"))
merged_data=[]
for file in files: 
    with open(file, 'r') as file:
        json_data = simplejson.load(file)
        df=pd.DataFrame(json_data)
        merged_data.append(df)

joined_data=pd.concat(merged_data)

tsvfiles = glob.glob(os.path.join(folder_path, "*questions.tsv*"))
merged_data=[]
for file in tsvfiles: 
    df=pd.read_csv(file,sep='\t',names=['question','answer'])
    merged_data.append(df)
joined_data=pd.concat(merged_data)
print(f'joined_data dataset size: {joined_data.shape}')

old_question = joined_data

processed_old_question = [utils.default_process(question) for question in old_question['question']]
choices_dict = {idx: el for idx, el in enumerate(processed_old_question)}

threshold = 80

def find_match(x):
    # limit=1 : extract the top1 match 
    match = process.extract(x, choices_dict, limit=1, scorer=rapidfuzz.fuzz.partial_ratio)[0]
    # filter by threshold 
    is_match = True if match[1]>threshold else False
    return is_match

sub_newoag = newoag.sample(n=1000, random_state=1, replace=False) 
# sub_newoag = newoag

tqdm.pandas(desc='apply')
sub_newoag['is_old'] = sub_newoag['title'].progress_apply(find_match)

print(f'filtered_newoag dataset size: {sub_newoag.shape}')
sub_newoag.index = sub_newoag['is_old']

# Filter by 'is_old'
filtered_newoag = sub_newoag[sub_newoag['is_old'] == False]
print(f'filtered_newoag dataset size: {filtered_newoag.shape}')
