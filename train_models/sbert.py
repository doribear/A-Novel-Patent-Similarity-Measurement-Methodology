from transformers import BertModel, BertTokenizer

import torch
import torch.nn as nn

class Sentence_Bert(nn.Module):
    def __init__(self) -> None:
        super(Sentence_Bert, self).__init__()

        self.device = 'cuda'
        
        self.bert = BertModel.from_pretrained('anferico/bert-for-patents')
        self.bert = self.bert.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('anferico/bert-for-patents')

        self.split_code = lambda x : [x[0], x[1:3], x[3]]
        self.bert.eval()


    def frozen(self):
        for p in self.bert.parameters():
            p.requires_grad = False
        print("Trnsformer's parameter all frozen")

    def encode(self, ids, attention_mask = None, type_ids = None):
        if type_ids == None:
            type_ids = torch.zeros(ids.shape, device = ids.device).long()
        if attention_mask == None:
            attention_mask = torch.ones(ids.shape, device = ids.device).long()
        embed = self.bert(ids, attention_mask, type_ids)
        return embed

    def get_document_similarity(self, documents : dict):
        """
        sentence bert를 활용해 document간 코사인 유사도를 출력하는 함수

        document format
        {"documents_0" : [["the", "dog", "chase" ,"a", "duck"], ["the" "cat" "chase" "a" "duck"], ["bibimbab"]], 
        "documents_1" : ["the", "dog", "chase" ,"a", "duck"], ["the", "dog", "chase" ,"a", "duck"], ["bibimbab"]]}
        """
        for key in documents.keys():
            #documents[key] = self.tokenizer(documents[key], return_tensors = 'pt', padding = 'max_length', is_split_into_words = True, add_special_tokens = False, truncation = True, max_length = 512).to(self.device)
            documents[key] = self.tokenizer(documents[key], return_tensors = 'pt', padding = 'max_length', is_split_into_words = False, add_special_tokens = True, truncation = True, max_length = 512).to(self.device)
        x, x_attention_mask, x_type_ids = documents['documents_0'].input_ids, documents['documents_0'].attention_mask, documents['documents_0'].token_type_ids
        y, y_attention_mask, y_type_ids = documents['documents_1'].input_ids, documents['documents_1'].attention_mask, documents['documents_1'].token_type_ids
        similiarity = self.forward(x, x_attention_mask, x_type_ids, y, y_attention_mask, y_type_ids)
        return similiarity

    def ipc_similarity(self, documents):
        """
        ipc code를 활용해 document간 코사인 유사도를 출력하는 함수

        document format
        {"documents_0" : [["H01L", "H01L", "H01L", "H01L"], ["H01L", "H01L", "H01L", "H01L"]], 
        "documents_1" : [["H01L"], ["H01L", "H01L", "H01L", "H01L"]]}
        """
        documents[0], documents[1] = documents['documents_0'], documents['documents_1']
        similarities = list(map(lambda x, y : len(set(x) & set(y)) / len(set(x) | set(y)), documents[0], documents[1]))
        similarities = torch.Tensor(similarities).to(self.device)
        return similarities
    
    def get_similarity(self, documents, ipc_codes):
        """
        ipc code 및 sentence bert의 결과를 토대로 최종 similarity를 도출

        ##documents format##
        {"documents_0" : [["the", "dog", "chase" ,"a", "duck"], ["the" "cat" "chase" "a" "duck"], ["bibimbab"]], 
        "documents_1" : ["the", "dog", "chase" ,"a", "duck"], ["the", "dog", "chase" ,"a", "duck"], ["bibimbab"]]}

        ##ipc codes##
        {"documents_0" : [["H01L", "H01L", "H01L", "H01L"], ["H01L", "H01L", "H01L", "H01L"]], 
        "documents_1" : [["H01L"], ["H01L", "H01L", "H01L", "H01L"]]}
        """
        cosine_similarity = self.get_document_similarity(documents)
        technical_similarity = self.ipc_similarity(ipc_codes)
        semanctic_distance = ((technical_similarity + 1) * cosine_similarity) / 2 # 0.7565
        return semanctic_distance
    
    def forward(
        self, 
        x : torch.LongTensor = None, 
        x_attention_mask : torch.LongTensor = None, 
        x_type_ids : torch.LongTensor = None, 
        y : torch.LongTensor = None, 
        y_attention_mask : torch.LongTensor = None,
        y_type_ids : torch.LongTensor = None):

        x, y = self.encode(x, x_attention_mask, x_type_ids), self.encode(y, y_attention_mask, y_type_ids)
        x, y = torch.mean(x.last_hidden_state, dim = 1), torch.mean(y.last_hidden_state, dim = 1)
        sim = torch.cosine_similarity(x, y, dim = 1)
        return sim