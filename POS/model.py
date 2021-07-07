import torch
from torch import nn
from torchcrf import CRF
from parameteres import Parameter as pm
'''
关于CRF： https://pytorch-crf.readthedocs.io/en/stable/#
'''
class BiLstmCRF(nn.Module):
    def __init__(self,embedding):
        super(BiLstmCRF,self).__init__()
        self.embedding = nn.Embedding(pm.vocab_size, pm.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        # ???改
        self.bilstm = nn.LSTM(pm.embedding_dim,pm.hidden_dim // 2,num_layers=1, bidirectional=True)    #bilstm 层
        self.hidden2tag = nn.Linear(pm.hidden_dim,pm.num_tags)     #线性层
        self.crf = CRF(pm.num_tags)  #crf 层
        self.dropout = nn.Dropout(0.5)
    def forward(self,x,y,mask):
        '''

        :param x: [maxseq,batchsize]
        :param y: [seq_length, batch_size]
        :param mask: [seq_length, batch_size]
        :return:
        '''
        # step1: Get the emission scores from the BiLSTM
        #
        '''Examples::
        >> > rnn = nn.LSTM(10, 20, 2)                   
        >> > input = torch.randn(5, 3, 10)
        >> > h0 = torch.randn(2, 3, 20)
        >> > c0 = torch.randn(2, 3, 20)
        >> > output, (hn, cn) = rnn(input, (h0, c0))  # h0,c0 为初始状态,不提供默认为0  input (seq_len, batch, input_size),
        '''
        embeds = self.embedding(x)      #[maxseq,batchsize] -> [maxseq,batchsize,embdeding_dim]
        lstm_out, self.hidden = self.bilstm(embeds)  #[maxseq,batchsize,embdeding_dim]  -> [maxseq,batchsize,hidden_dim]
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.hidden2tag(lstm_out)   #[maxseq,batchsize,hidden_dim] ->[maxseq,batchsize,num_tags]

        #step2 :cmpute cmpute loss and crf
        loss = - self.crf(lstm_out,y,mask)
        pred = self.crf.decode(lstm_out,mask)
        return loss,pred
    def decode(self,x,mask):
        embeds = self.embedding(x)  # [maxseq,batchsize] -> [maxseq,batchsize,embdeding_dim]
        lstm_out, self.hidden = self.bilstm(
            embeds)  # [maxseq,batchsize,embdeding_dim]  -> [maxseq,batchsize,hidden_dim]
        lstm_out = self.hidden2tag(lstm_out)  # [maxseq,batchsize,hidden_dim] ->[maxseq,batchsize,num_tags]
        pred = self.crf.decode(lstm_out, mask)
        return pred
