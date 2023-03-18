from train_models.sbert import Sentence_Bert
from train_models.sim_lstm import Siamese_LSTM_Model
from train_models.sim_bilstm import Siamese_BiLSTM_Model
from train_models.sim_cnn import Siamese_CNN_Model
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import pandas as pd
import ast

import numpy as np

# bert
sbert = Sentence_Bert().eval()

# other model
lstm = Siamese_LSTM_Model().eval()
bilstm = Siamese_BiLSTM_Model().eval()
rawcnn = Siamese_CNN_Model().eval()

lstm.load_state_dict(torch.load("", "cpu"))
bilstm.load_state_dict(torch.load("", "cpu"))
rawcnn.load_state_dict(torch.load("", "cpu"))

# data prepare
data = pd.read_excel("", engine = "openpyxl")

A_codes = list(map(lambda x : ast.literal_eval(x), data['code']))
B_codes = list(map(lambda x : ast.literal_eval(x), data['target_ipc']))

A_titles_abstracts = data['meta'].tolist()
B_titles_abstracts = data['target_meta'].tolist()


def get_sbert_similarity(A_codes = None, B_codes = None, A_titles_abstracts = None, B_titles_abstracts = None, method : str = "proposed"): # method = ["proposed", "SD", "TD"]
    similarities = []
    for a_code, b_code, a_title_abstract, b_title_abstract in tqdm(zip(A_codes, B_codes, A_titles_abstracts, B_titles_abstracts), total = len(A_codes), desc = "calculating sbert similarity"):
        tile_abstract = {"documents_0" : [a_title_abstract], "documents_1" : [b_title_abstract]}
        ipc_document = {"documents_0" : [a_code], "documents_1" : [b_code]}
        if method == "proposed":
            similarity = sbert.get_similarity(tile_abstract, ipc_document)
        elif method == "SD":
            similarity = sbert.get_document_similarity(tile_abstract)
        elif method == "TD":
            similarity = sbert.ipc_similarity(ipc_document)
        similarities.append(similarity.item())
        torch.cuda.empty_cache()
    return similarities



def get_other_similaritiy(model): # model = ["lstm", "bilstm", "cnn"]
    model = model.to("cuda")
    similarities = []
    for a_title_abstract, b_title_abstract in tqdm(zip(A_titles_abstracts, B_titles_abstracts), total = len(A_titles_abstracts), desc = "calculating similarity"):
        a_title_abstract = sbert.tokenizer(a_title_abstract, return_tensors = 'pt', padding = 'max_length', is_split_into_words = False, add_special_tokens = True, truncation = True, max_length = 512).to("cuda")
        b_title_abstract = sbert.tokenizer(b_title_abstract, return_tensors = 'pt', padding = 'max_length', is_split_into_words = False, add_special_tokens = True, truncation = True, max_length = 512).to("cuda")
        a_title_abstract, b_title_abstract = a_title_abstract.input_ids, b_title_abstract.input_ids
        similarity = model(a_title_abstract, b_title_abstract).item()
        torch.cuda.empty_cache()
        similarities.append(similarity)
    return similarities


data['proposed'] = get_sbert_similarity(A_codes, B_codes, A_titles_abstracts, B_titles_abstracts, "proposed")
data['Bert'] = get_sbert_similarity(A_codes, B_codes, A_titles_abstracts, B_titles_abstracts, "SD")

data['lstm_similarity'] = get_other_similaritiy(lstm)
data['bilstm_similarity'] = get_other_similaritiy(bilstm)
data['cnn_similarity'] = get_other_similaritiy(rawcnn)

comparisons = ['proposed', 'Bert', 'lstm_similarity', 'bilstm_similarity', 'cnn_similarity']
results = {}
pvalues = {}

for key in comparisons:
    results[f'{key}_pearson'] = pearsonr(np.array(data[key]), np.array(data['similarity'])).statistic
    results[f'{key}_spearman'] = spearmanr(np.array(data[key]), np.array(data['similarity'])).statistic

    pvalues[f'{key}_pearson'] = pearsonr(np.array(data[key]), np.array(data['similarity'])).pvalue
    pvalues[f'{key}_spearman'] = spearmanr(np.array(data[key]), np.array(data['similarity'])).pvalue

experiment_result = pd.DataFrame({'model' : results.keys(), 'cor' : results.values()})
experiment_result_pvalue = pd.DataFrame({'model' : pvalues.keys(), 'cor' : pvalues.values()})

data.to_excel('', index = False)
experiment_result_pvalue.to_excel('', index = False)
experiment_result.to_excel('', index = False)
