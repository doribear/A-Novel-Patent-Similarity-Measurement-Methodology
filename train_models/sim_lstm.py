import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(39859, 128, padding_idx = 0)
        self.bilstm = nn.LSTM(128, 2048, bidirectional = False)
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        out = self.fc(output)
        return out
    

class Siamese_LSTM_Model(nn.Module):
    def __init__(self) -> None:
        super(Siamese_LSTM_Model, self).__init__()
        self.padding_idx = 0
        self.LSTM = LSTM_Model()
        
    def forward(self, x, y, normalize = False):
        x_mask, y_mask = (x != self.padding_idx).long(), (y != self.padding_idx).long()
        x_bunmo, y_bunmo = torch.sum(x_mask, -1), torch.sum(y_mask, -1)
        x, y = self.LSTM(x) * x_mask[:, :, None], self.LSTM(y) * y_mask[:, :, None]
        x, y = torch.sum(x, dim = 1) / x_bunmo[:, None], torch.sum(y, dim = 1) / y_bunmo[:, None]
        sim = torch.cosine_similarity(x, y, dim = 1)
        if normalize:
            sim = (sim + 1) / 2
        return sim