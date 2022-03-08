import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=4)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden) 
        x = x.contiguous().view(-1, 128) 
        x = self.sigmoid(self.fc(x))
        return x[-1], hidden
        
        
        
def init_hidden(batch_size, device):
    h0 = torch.zeros((4,batch_size,128)).to(device)
    c0 = torch.zeros((4,batch_size,128)).to(device)
    hidden = (h0,c0)
    return hidden