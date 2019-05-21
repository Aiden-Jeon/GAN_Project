# DCGAN.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from math import log2, sqrt
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, input_size, hidden_dim, dtype):
        super(Generator, self).__init__()
        d_num = int(log2(input_size))
        features = []
        for after_dim in reversed(range(d_num-2)):
            if after_dim == d_num-3:
                after_dim = 2 ** after_dim
                features += [
                    nn.ConvTranspose2d(latent_dim, hidden_dim * after_dim, 4, 1, 0),
                    nn.BatchNorm2d(hidden_dim * after_dim),
                    nn.ReLU(True)]
            else:
                after_dim = 2 ** after_dim        
                features += [
                    nn.ConvTranspose2d(hidden_dim * before_dim, hidden_dim * after_dim, 4, 2, 1),
                    nn.BatchNorm2d(hidden_dim * after_dim),
                    nn.ReLU(True)]
            before_dim = after_dim
        features += [
            nn.ConvTranspose2d(hidden_dim, 3, 4, 2, 1),
            nn.Tanh()]
        
        self.features = nn.Sequential(*features).type(dtype)
        self.latent_dim = latent_dim      
        
    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        return self.features(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, dtype):
        super(Discriminator, self).__init__()
        d_num = int(log2(input_size))
        features = []
        for after_dim in range(d_num-2):
            if after_dim == 0:
                after_dim = 2 ** after_dim
                features += [
                    nn.Conv2d(3, hidden_dim * after_dim, 4, 2, 1),
                    nn.ReLU(True)]
            else:
                after_dim = 2 ** after_dim        
                features += [
                    nn.Conv2d(hidden_dim * before_dim, hidden_dim * after_dim, 4, 2, 1),
                    nn.BatchNorm2d(hidden_dim * after_dim),
                    nn.ReLU(True)]
            before_dim = after_dim
        features += [
            nn.Conv2d(hidden_dim * after_dim, 1, 4, 1, 0),
            nn.Sigmoid()]

        self.features = nn.Sequential(*features).type(dtype)
        
    def forward(self, x):
        return self.features(x).squeeze().view(-1,1)


class trainer:
    def __init__(self, D, D_optimizer, G, G_optimizer, dtype):
        self.D = D
        self.D_optimizer = D_optimizer
        self.G = G
        self.G_optimizer = G_optimizer
        self.D_loss_hist = []
        self.G_loss_hist = []
        self.dtype = dtype

    def show_img(self, num_img, latent_dim):
        H = sqrt(num_img)
        W = sqrt(num_img)
        if H*W < num_img:
            W += 1
        self.G.eval()
        z = Variable(torch.randn((num_img, latent_dim)).type(self.dtype))
        fake_data = self.G(z)
        fake_data = fake_data.permute(0,2,3,1)
        fake_data = fake_data.cpu().data
        for i, x in enumerate(fake_data):
            plt.subplot(H, W, i+1)
            plt.imshow((x+1)/2)
        plt.show()
    
    def train(self, num_epochs, data_loader, loss_fn, print_every, num_img):
        for epoch in range(num_epochs):
        	print('Starting %d/%d' % (epoch+1, num_epochs+1))
            self.G.train()
            self.D.train()
            total_G_loss = 0
            total_D_loss = 0
            for t, (real_data, _) in enumerate(data_loader):
                batch_size = real_data.size(0)
                real_data = Variable(real_data.type(self.dtype))

                target_real = Variable(torch.ones(batch_size, 1).type(self.dtype))
                target_fake = Variable(torch.zeros(batch_size, 1).type(self.dtype))

                D_result_from_real = self.D(real_data)
                D_loss_real = loss_fn(D_result_from_real, target_real)

                z = Variable(torch.randn((batch_size, self.G.latent_dim)).type(self.dtype))
                fake_data = self.G(z)
                D_result_from_fake = self.D(fake_data)
                D_loss_fake = loss_fn(D_result_from_fake, target_fake)

                D_loss = D_loss_real + D_loss_fake
                total_D_loss += D_loss.data

                self.D.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

                z = Variable(torch.randn((batch_size, self.G.latent_dim)).type(self.dtype))
                fake_data = self.G(z)

                D_result_from_fake = self.D(fake_data)
                G_loss = loss_fn(D_result_from_fake, target_real)
                total_G_loss += G_loss.data

                self.G.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

            if (epoch+1) % print_every == 0:
                print('D loss %.4f, G loss %.4f' % (total_D_loss/(t+1), total_G_loss/(t+1)))
                self.show_img(num_img, self.G.latent_dim)

            self.D_loss_hist += [total_D_loss/(t+1)]
            self.G_loss_hist += [total_G_loss/(t+1)]
