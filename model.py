import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np


class BidirectionalLSTM_front(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM_front, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input, residual=False):
        recurrent, _ = self.rnn(input)
        if residual:
            recurrent = recurrent + input
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = recurrent.view(T, b, -1)  # [T, b, nOut]

        return output

class BidirectionalLSTM_backend(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM_backend, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input, residual=False):
        recurrent, _ = self.rnn(input)
        if residual:
            recurrent = recurrent + input
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)  # [T, b, nOut]

        return output
class ASR_CTC(nn.Module):

    def __init__(self, nclass=33, nh=256, nOut=512, leakyRelu=False):
        super(ASR_CTC, self).__init__()
        
        self.conv1 = nn.Conv1d(13, 20, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(20, 36, kernel_size=2, stride=1, padding=1)
        #self.conv3 = nn.Conv1d(36, 72, kernel_size=2, stride=1)
        #self.conv4 = nn.Conv1d(72, 96, kernel_size=3, stride=1)
        #self.conv5 = nn.Conv1d(96, 144, kernel_size=4, stride=1)
        #self.conv6 = nn.Conv1d(144, 144, kernel_size=3, stride=1)
        
        self.rnn_front = BidirectionalLSTM_front(36, nh, nh)
        self.rnn_backend = BidirectionalLSTM_backend(nh*2, nh, nclass)


    def forward(self, input):
        # rnn features
        #print(input.size())
        input = input.permute(0,2,1)
        
        output = self.conv1(input)
        output = self.conv2(output)
        #output = self.conv3(output)
        #output = self.conv4(output)
        #output = self.conv5(output)
        #output = self.conv6(output)
        #print(output.size())
        output = output.permute(0,2,1)
        #print(output.size())
        
        output = self.rnn_front(output, residual=False)   # Take (CNN output + RNN output) as input of embedding layer
        #print(output.size())
        #output = output.permute(0,2,1)
        output = self.rnn_backend(output)  # [w, b, nclass]
        #print(output.size())
        return output
        #print(output.size())

'''
if __name__ == "__main__":
    x = torch.rand(32, 200, 13).cuda()
    net = ASR_CTC(28, 256, 512)
    #net = BidirectionalLSTM_front(13, 20, 256, 512)
    net.cuda()
    y = net(x)
    #print(y.size())
'''