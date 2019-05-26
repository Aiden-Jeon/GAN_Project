# Pix2Pix.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

from math import sqrt, log2
import time

class ConvUp(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, batch_norm, dtype):
        super(ConvUp, self).__init__()
        features = [nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding)]
        if batch_norm:
            features += [nn.BatchNorm2d(output_nc)]
#         features += [nn.LeakyReLU(0.2, True)]
        features += [nn.ReLU(True)]
        self.features = nn.Sequential(*features).type(dtype)

    def forward(self, x):
        return self.features(x)
    

class DeconvDown(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, batch_norm, dropout, dtype):
        super(DeconvDown, self).__init__()
        features = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride, padding)]
        if batch_norm:
            features += [nn.BatchNorm2d(output_nc)]
        if dropout:
            features += [nn.Dropout2d(0.5)]
        features += [nn.ReLU(True)]
        self.features = nn.Sequential(*features).type(dtype)

    def forward(self, x, eo):
        x = torch.cat([x, eo], 1)
        return self.features(x)


class Genereator(nn.Module):
    def __init__(self, input_size, d, dtype):
        super(Genereator, self).__init__()
        self.dtype = dtype
        # Encoder
        self.encoder = [ConvUp(3, d, 4, 2, 1, False, dtype)]
        for i in range(int(log2(input_size))-2):
            if i < 3:
                self.encoder += [ConvUp(d, d*2, 4, 2, 1, True, dtype)]
                d *= 2
            else:
                self.encoder += [ConvUp(d, d, 4, 2, 1, True, dtype)]
        self.encoder_last = nn.Sequential(
            nn.Conv2d(d, d, 4, 2, 1),
            nn.ReLU(True)).type(dtype)
        
        # Decoder
        self.decoder = [DeconvDown(d, d, 4, 2, 1, True, True, dtype)]
        for i in range(int(log2(input_size))-2):
            if i < 2:
                self.decoder += [DeconvDown(d*2, d, 4, 2, 1, True, True, dtype)]
            elif i >= 2 and i <log2(input_size)-5:
                self.decoder += [DeconvDown(d*2, d, 4, 2, 1, True, False, dtype)]
            else:
                d = int(d/2)
                self.decoder += [DeconvDown(d*2*2, d, 4, 2, 1, True, False, dtype)]
        self.decoder_last = nn.Sequential(
            nn.ConvTranspose2d(d*2, 3, 4, 2, 1),
            nn.Tanh()).type(dtype)
        
    def forward(self, x):
        encoder_output = []
        # Enocder
        for layer in self.encoder:
            x = layer(x)
            encoder_output += [x]
        x = self.encoder_last(x)
        
        # Decoder
        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = layer(x, torch.Tensor([]).type(self.dtype))
            else:
                x = layer(x, encoder_output[-i])
        x = torch.cat([x, encoder_output[0]], 1)
        return self.decoder_last(x)
    

class Discriminator(nn.Module):
    def __init__(self, d, dtype):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, d, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            nn.Conv2d(d*4, d*8, 4, 1, 1),
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            nn.Conv2d(d*8, 1, 4, 1, 1),
            nn.Sigmoid()).type(dtype)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        return self.features(x)

class Trainer:
    def __init__(self, D, D_optimizer, G, G_optimizer, l1_lambda, dtype):
        self.D = D
        self.G = G
        self.D_optimizer = D_optimizer
        self.G_optimizer = G_optimizer
        self.D_loss_hist = []
        self.G_loss_hist = []
        self.dtype = dtype
        self.bce_loss_fn = nn.BCELoss().type(dtype)
        self.l1_loss_fn = nn.L1Loss().type(dtype)
        self.l1_lambda = l1_lambda
        
    def show_img(self, x, y):
        H = sqrt(x.size()[0])
        W = sqrt(x.size()[0])
        if H*W < x.size()[0]:
            W += 1
        self.G.eval()
        result_x = (self.G(x)+1)/2
        result_y = (y+1)/2
        result = torch.cat([result_x, result_y], 3).cpu().permute(0,2,3,1).data
        for i, x in enumerate(result):
            plt.subplot(H, W, i+1)
            plt.imshow(x)
        plt.show()    
    
    def train(self, num_epochs, data_loader, print_every, num_img):
        for epoch in range(num_epochs):
            tic = time.time()
            print('Starting %d/%d' % (epoch+1, num_epochs))
            self.G.train()
            self.D.train()
            total_G_loss = 0
            total_D_loss = 0
            for x,y in data_loader:
                x = Variable(x.type(self.dtype))
                y = Variable(y.type(self.dtype))

                # train D
                D_result = self.D(x, y).squeeze()
                D_real = Variable(torch.ones(D_result.size())).type(self.dtype)
                D_real_loss = self.bce_loss_fn(D_result, D_real)

                G_result = self.G(x)
                D_result = self.D(x, G_result).squeeze()
                D_fake = Variable(torch.zeros(D_result.size()).type(self.dtype))
                D_fake_loss = self.bce_loss_fn(D_result, D_fake)

                D_loss = (D_real_loss + D_fake_loss) * 0.5
                total_D_loss += D_loss.cpu().data

                self.D.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

                # train G
                G_result = self.G(x)
                D_result = self.D(x, G_result).squeeze()

                G_loss = self.bce_loss_fn(D_result, D_real) 
                G_loss += self.l1_lambda * self.l1_loss_fn(G_result, y)
                total_G_loss += G_loss.cpu().data

                self.G.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()


            if (epoch+1) % print_every == 0:
                toc = time.time()
                print('Time: %.4f, D loss %.4f, G loss %.4f' % 
                      (toc-tic, total_D_loss/len(data_loader), total_G_loss/len(data_loader)))
                self.show_img(x[:num_img], y[:num_img])

            self.D_loss_hist += [total_D_loss/len(data_loader)]
            self.G_loss_hist += [total_G_loss/len(data_loader)]