import pandas as pd
import simplejson
from tqdm.auto import tqdm
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import numpy as np

# Load dataset

newoag=pd.read_json('oag_qa_20230512.json',lines=True)
# newoag_title=newoag[['title','answers']]
print(f'newoag_title dataset size: {newoag.shape}')

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

# Transform text to vectors with TF-IDF
newoag_title = newoag['title']
vectorizer = TfidfVectorizer(min_df=1)
tf_idf_matrix_newoag = vectorizer.fit_transform(newoag_title) 
print(tf_idf_matrix_newoag.shape)

old_question_title = old_question['question']
tf_idf_matrix_old_question = vectorizer.fit_transform(old_question_title)
print(tf_idf_matrix_old_question.shape)

# Compute Cosine similarity
def cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32
    nnz_max = M*ntop    
    
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)
    return csr_matrix((data,indices,indptr),shape=(M,N))

matches = cossim_top(tf_idf_matrix_newoag, tf_idf_matrix_old_question.transpose(), 10, 0.8)

print(matches.shape)
# Create a match table

# def get_matches_df(sparse_matrix, name_vector, top=100):
#     non_zeros = sparse_matrix.nonzero()
    
#     sparserows = non_zeros[0]
#     sparsecols = non_zeros[1]
    
#     if top:
#         nr_matches = top
#     else:
#         nr_matches = sparsecols.size
    
#     left_side = np.empty([nr_matches], dtype=object)
#     right_side = np.empty([nr_matches], dtype=object)
#     similairity = np.zeros(nr_matches)
    
#     for index in range(0, nr_matches):
#         left_side[index] = name_vector[sparserows[index]]
#         right_side[index] = name_vector[sparsecols[index]]
#         similairity[index] = sparse_matrix.data[index]
    
#     return pd.DataFrame({'title': left_side,
#                           'similar_title': right_side,
#                            'similairity_score': similairity})

# matches_df = pd.DataFrame()
# matches_df = get_matches_df(matches, newoag['title'], top=10000)
# # Remove all exact matches
# matches_df = matches_df[matches_df['similairity_score'] < 0.99999] 
# print(matches_df.sample(10))