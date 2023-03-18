import torch
import torch.nn as nn

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        
        self.embedding = nn.Embedding(39859, 128, padding_idx = 0)
        self.conv1 = nn.Sequential(nn.Conv1d(128, 512, 3, 1, 1), nn.MaxPool1d(3, 1, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(512, 2048, 3, 1, 1), nn.MaxPool1d(3, 1, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(2048, 512, 3, 1, 1), nn.MaxPool1d(3, 1, 1), nn.ReLU())
        self.fcn = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text):
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]

        out = self.conv1(embedded)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fcn(out.permute(0, 2, 1))
        output = self.dropout(out)
        return output

class Siamese_CNN_Model(nn.Module):
    def __init__(self) -> None:
        super(Siamese_CNN_Model, self).__init__()
        self.padding_idx = 0
        self.CNN = CNN_Model()
        
    def forward(self, x, y, normalize = False):
        x_mask, y_mask = (x != self.padding_idx).long(), (y != self.padding_idx).long()
        x_bunmo, y_bunmo = torch.sum(x_mask, -1), torch.sum(y_mask, -1)
        x, y = self.CNN(x) * x_mask[:, :, None], self.CNN(y) * y_mask[:, :, None]
        x, y = torch.sum(x, dim = 1) / x_bunmo[:, None], torch.sum(y, dim = 1) / y_bunmo[:, None]
        sim = torch.cosine_similarity(x, y, dim = 1)
        if normalize:
            sim = (sim + 1) / 2
        return sim

