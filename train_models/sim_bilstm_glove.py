import torch
import torch.nn as nn

class BiLSTM_Model(nn.Module):
    def __init__(self):
        super(BiLSTM_Model, self).__init__()
        self.bilstm = nn.LSTM(200, 1024, bidirectional = True)
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        output, _ = self.bilstm(x)
        output = output.nan_to_num()
        out = self.fc(output)
        return out.nan_to_num()
    

class Siamese_BiLSTM_glove_Model(nn.Module):
    def __init__(self) -> None:
        super(Siamese_BiLSTM_glove_Model, self).__init__()
        self.padding_idx = 0
        self.LSTM = BiLSTM_Model()
        
    def forward(self, x, y, normalize = False):
        x_mask, y_mask = torch.sum(x, dim = -1).long(), torch.sum(y, dim = -1).long()
        x_mask, y_mask = (x_mask != self.padding_idx).long(), (y_mask != self.padding_idx).long()
        x_bunmo, y_bunmo = torch.sum(x_mask, -1), torch.sum(y_mask, -1)
        x, y = self.LSTM(x) * x_mask[:, :, None], self.LSTM(y) * y_mask[:, :, None]
        x, y = torch.sum(x, dim = 1) / x_bunmo[:, None], torch.sum(y, dim = 1) / y_bunmo[:, None]
        sim = torch.cosine_similarity(x, y, dim = 1)
        if normalize:
            sim = (sim + 1) / 2
        sim = sim.nan_to_num()
        return sim