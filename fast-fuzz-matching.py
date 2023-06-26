import pandas as pd
import simplejson
from tqdm.auto import tqdm
import os
import glob
import re
from ftfy import fix_text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import time

# Load dataset

t0 = time.time()

newoag=pd.read_json('oag_qa_20230512.json',lines=True)
# newoag_title=newoag[['title','answers']]
print(f'newoag dataset size: {newoag.shape}')

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

old_question = joined_data['question'].unique()
print(f'unique old_question dataset size: {old_question.shape}')

print('--------------------------------------------------')
print('Vecorizing the data - this could take a few minutes for large datasets...')

# clean strings
def ngrams(string, n=4):
    string = str(string)
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# Bulid custom vectorizer with string clean analyzer 
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(old_question)
print('Vecorizing completed!')

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)

title_column = 'title'
sub_newoag = newoag[:1000] 
# sub_newoag = newoag
newoag_title = sub_newoag[title_column].values
print(f'newoag_title dataset size: {newoag_title.shape}')
# newoag_title = newoag['title'].sample(n=1000, random_state=1, replace=False) 

# matching query
def getNerestN(query):
    queryTFIDF_ = vectorizer.transform(query)
    distance, indices = nbrs.kneighbors(queryTFIDF_)
    return distance, indices

print('--------------------------------------------------')
print('geting nearest n...')
start_time = time.time()

distances, indices = getNerestN(newoag_title)

end_time = time.time()
print(f"finished in {end_time-start_time} seconds")

# find matches
print('--------------------------------------------------')
print('find matches...')

matches = []
for i, j in enumerate(indices):
    temp = [round(distances[i][0], 2), old_question[j][0], newoag_title[i]]
    matches.append(temp)



print('--------------------------------------------------')
print("Building datafram...")
# Match confidence : smaller is better
matches = pd.DataFrame(matches, columns=['Match confidence','old question','newoag_title'])
print("Done")

# Merge Dataframe
newoag_with_matches = pd.concat([sub_newoag, matches], axis=1)
print(newoag_with_matches[['Match confidence','old question',  'title']])

# filter by mathc confidence
threshold = 0.5
filtered_newoag = newoag_with_matches[newoag_with_matches['Match confidence'] <= threshold]
print(filtered_newoag[['Match confidence','old question',  'title']])

t1 = time.time()
print(f"finished fast fuzz matching in {t1-t0} seconds")


