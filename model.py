""" Written by Youval Vanlaer
13/03/2021 """

import torch
import torch.jit as jit
from torch import nn
from torch import optim
from termcolor import colored
import sys
import os

from dataloader import MAPSDataLoader

class ConvNet(nn.Module):
    def __init__(self, window_size, output_size):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 50, (5,25))
        self.conv2 = nn.Conv2d(50, 50, (3, 5))
        self.maxpool = nn.MaxPool2d((1,3))
        self.fc1 = nn.Linear(1200, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.window_size = window_size

    def forward(self, x):
        x = x.view((-1, 1, self.window_size, 252))
        x = self.tanh(self.conv1(x))
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.tanh(self.conv2(x))
        x = self.dropout(x)
        x = self.maxpool(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.sigmoid(self.fc1(x))
        x = self.dropout(x)

        x = self.sigmoid(self.fc2(x))
        x = self.dropout(x)

        x = self.sigmoid(self.fc3(x))

        return x


class RNN_Nade(nn.Module):
    def __init__(self, input_size, hidden_size_rnn, hidden_size_nade):
        super(RNN_Nade, self).__init__()

        self.hidden_size_rnn = hidden_size_rnn # 200
        self.hidden_size_nade = hidden_size_nade # 150
        self.input_size = input_size
        self.fc_link1 = nn.Linear(200, 1, bias=True)
        self.fc_link2 = nn.Linear(200, 150, bias=True)
        self.weight_h = nn.Parameter(torch.randn(input_size, hidden_size_nade))
        self.weight_p = nn.Parameter(torch.randn(input_size, hidden_size_nade))
        self.cell = RNN_Nade_Cell(input_size, hidden_size_rnn)
        self.sigmoid = nn.Sigmoid()

    #@jit.script_method
    def nade_likelihood(self, out, x, is_sampling = False):
        b1 = self.fc_link1(out)
        b2 = self.fc_link2(out)

        p = torch.zeros(self.input_size)
        p[0] = self.sigmoid(torch.matmul(self.weight_p.t()[:,0],self.sigmoid(b2)) + b1)
        wi_xi = torch.zeros(self.hidden_size_nade)
        for i in range(1, self.input_size):
            wi_xi = wi_xi + self.weight_h.t()[:,i - 1] * x[i - 1]
            hi = self.sigmoid(wi_xi + b2)
            p[i] = self.sigmoid(torch.matmul(self.weight_p.t()[:,i],hi) + b1)

        return p

    #@jit.script_method
    def forward(self, inputs):
        log_likelihood = 0
        out = torch.randn(self.hidden_size_rnn)
        for t in range(len(inputs)):
            out = self.cell(inputs[t], out)

            p = self.nade_likelihood(out, inputs[t])
            log_likelihood += torch.log(p).sum()

        return log_likelihood

    #@jit.script_method
    def sample(self, posteriori):
        out = torch.randn(self.hidden_size_rnn)
        for t in range(len(inputs)):
            out = self.cell(posteriori[t], out)

            p = nade_likelihood(out, None, is_sampling = True)


class RNN_Nade_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN_Nade_Cell, self).__init__()

        self.weight_lf = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_lr = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.tanh = nn.Tanh()

    #@jit.script_method
    def forward(self, input, hidden):
        output = torch.matmul(input, self.weight_lf.t()) + self.bias + torch.matmul(hidden, self.weight_lr.t())

        return self.tanh(output)

if __name__ == '__main__':
    model_conv = ConvNet(window_size=7, output_size=88)
    model_nade = RNN_Nade(input_size=88, hidden_size_rnn=200, hidden_size_nade=150)
    dataLoader = MAPSDataLoader('preprocess/AkPnBcht/MUS/')

    loss = nn.BCELoss()
    lr = 0.01
    start_epoch = 0
    end_epoch = 10
    optimizer = optim.SGD(model_conv.parameters(), lr=lr, momentum=0.9)

    for epoch in range(start_epoch, end_epoch):
        print('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        model_conv.train()
        total_loss = 0
        avg_loss = 0
        for i, batch in enumerate(iter(dataLoader)):
            optimizer.zero_grad()
            input, target = batch
            output = loss(model_conv(input), target)
            output.backward()
            optimizer.step()
            total_loss += output.detach()
            total_loss = total_loss / (i + 1)
            sys.stdout.write(
                "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}".format(
                    progress=colored("{:.3f}".format(epoch + i / len(dataLoader)), "green", attrs=['bold']),
                    ce_loss=colored("{:.4f}".format(output.item()), "blue", attrs=['bold']),
                    avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold'])))
            sys.stdout.flush()
        sys.stdout.write("\n")
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save("model/output"+str(epoch)+".pt", {'epoch': epoch, 'state_dict': model_conv.state_dict(), 'optimizer': optimizer.state_dict()})
            #loglikelihood = model_nade(output)
