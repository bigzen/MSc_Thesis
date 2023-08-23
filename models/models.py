import torch as T
import numpy as np

class RNN_head(T.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_head, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.fc0 = T.nn.Linear(input_size, int(hidden_size/2))
        self.lstm = T.nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        #self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = T.nn.Linear(hidden_size, int(hidden_size/2))
        self.relu = T.nn.ReLU()
        self.fc2 = T.nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc3 = T.nn.Linear(int(hidden_size/2), num_classes)
        #x_values = np.linspace(-1, 1, 5)
        #w_m = T.Tensor(np.exp(-np.power(x_values, 2.) / (2 * np.power(1, 2.))))
        #self.weights = T.unsqueeze(w_m/sum(w_m),0).expand(3,1,-1)
        #self.smooth = T.nn.Conv1d(3,3,[3,1,5],groups=3,padding='same')
    
    def forward(self, x):
        x = x.float()
        #out = self.relu(self.fc0(x))
        h0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda() 
        c0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda()
        out, _ = self.lstm(x, (h0,c0)) 
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        #out = T.transpose(out,0,1)
        #out = T.unsqueeze(out,0)
        #out=T.nn.functional.conv1d(out,self.weights.cuda(),groups=3,padding='same')
        #out=T.squeeze(out)
        #filt_out = T.transpose(out,0,1)
        return out

class RNN(T.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.fc0 = T.nn.Linear(input_size, int(hidden_size/2))
        self.lstm = T.nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        #self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = T.nn.Linear(int(hidden_size*1.5), int(hidden_size/2))
        self.relu = T.nn.ReLU()
        self.fc05 = T.nn.Linear(int(hidden_size), int(hidden_size*1.5))
        self.fc2 = T.nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc3 = T.nn.Linear(int(hidden_size/2), num_classes)
    
    def forward(self, x):
        x = x.float()
        #out = self.relu(self.fc0(x))
        h0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda() 
        c0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda()
        out, _ = self.lstm(x, (h0,c0))
        out = self.relu(self.fc05(out)) 
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out) 
        return out
    
class RNN2(T.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.fc0 = T.nn.Linear(input_size, int(hidden_size/2))
        self.lstm = T.nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        #self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = T.nn.Linear(int(hidden_size*1.5), int(hidden_size/2))
        self.relu = T.nn.ReLU()
        self.fc05 = T.nn.Linear(int(hidden_size), int(hidden_size*1.5))
        self.fc2 = T.nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc3 = T.nn.Linear(int(hidden_size/2), num_classes)
    
    def forward(self, x):
        x = x.float()
        #out = self.relu(self.fc0(x))
        h0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda() 
        c0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda()
        out, _ = self.lstm(x, (h0,c0))
        out = self.relu(self.fc05(out)) 
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out) 
        return out