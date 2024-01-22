import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class QNetwork(nn.Module):
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, step_size=1 ,
                 name='QNetwork'):
        super(QNetwork, self).__init__()

        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        # self.hidden_cell = (torch.zeros(-1, state_size, hidden_size), torch.zeros(-1, state_size, hidden_size))

    def forward(self, x):
        # print("x.type is : {}".format(x.dtype))
        lstm_out, self.hidden_cell = self.lstm(x)
        # print("hidden_cell.type is : {}".format(self.hidden_cell.dtype))
        lstm_out = lstm_out[:, -1, :]  # Extract the output at the last time step
        fc1_out = torch.relu(self.fc1(lstm_out))
        output = self.fc2(fc1_out)
        return output


from collections import deque

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size,step_size):
        idx = np.random.choice(np.arange(len(self.buffer)-step_size), 
                               size=batch_size, replace=False)
        
        res = []                       
                             
        for i in idx:
            temp_buffer = []  
            for j in range(step_size):
                temp_buffer.append(self.buffer[i+j])
            res.append(temp_buffer)
        return res    
        

