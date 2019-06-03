# CycleGAN.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import time


def resnet_init_weight(m, mean, std):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResnetBlock(nn.Module):
    def __init__(self, channel, kernel, stride, reflect_padding):
        super(ResnetBlock, self).__init__()
        self.features = nn.Sequential(
            nn.ReflectionPad2d(reflect_padding),
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel),
            nn.ReflectionPad2d(reflect_padding),
            nn.Conv2d(channel, channel, kernel, stride, 0),
            nn.InstanceNorm2d(channel)
        )
    
    def forward(self, x):
        x_res = self.features(x)
        return x + x_res


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, nb, dtype):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = nb
        
        # Encdoer 
        self.encoder = nn.Sequential(
            # Conv 1
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, 1, 0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # Conv 2
            nn.Conv2d(ngf, ngf*2, 3, 2, 1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            # Conv 3
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True)
        ).type(dtype)

        # Resnet 
        # 256x256 -> 9 / 128x128 -> 6
        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks += [ResnetBlock(ngf * 4, 3, 1, 1)]
            # self.resnet_blocks[i].apply(lambda x :resnet_init_weight(x, 0, 0.2))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks).type(dtype)
    
        # Decoder
        self.decocder = nn.Sequential(
            # Deconv1
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            # Deconv2
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, 1, 0),
            nn.Tanh()
        ).type(dtype)

    def forward(self, x):
        x = self.encoder(x)
        x = self.resnet_blocks(x)
        x = self.decocder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf, dtype):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(input_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            # Conv2
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            # Conv3
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            # Conv4
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            # Conv5
            nn.Conv2d(ndf*8, output_nc, 4, 1, 1)
        ).type(dtype)
        
    def forward(self, x):
        return self.features(x)


class ImagePool():
    ## https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py ##
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class Trainer:
    def __init__(self, D_A, D_A_optimizer, lambdaA, 
                 D_B, D_B_optimizer, lambdaB,
                 G_A, G_B, G_optimizer,  
                 dtype):
        self.D_A = D_A
        self.D_B = D_B
        self.G_A = G_A
        self.G_B = G_B
        
        self.D_A_optimizer = D_A_optimizer
        self.D_B_optimizer = D_B_optimizer
        self.G_optimizer = G_optimizer
        
        self.lambdaA = lambdaA
        self.lambdaB = lambdaB
        self.fakeA_store = ImagePool(50)
        self.fakeB_store = ImagePool(50)
        
        self.train_hist = {}
        
        self.BCE_loss = nn.BCELoss().type(dtype)
        self.MSE_loss = nn.MSELoss().type(dtype)
        self.L1_loss = nn.L1Loss().type(dtype)
        
        self.dtype = dtype
        
    def show_img(self, realA, realB):
        self.G_A.eval()
        self.G_B.eval()

        fakeB = self.G_A(realA)
        recA = self.G_B(fakeB)
        fakeA = self.G_B(realB)
        recB = self.G_A(fakeA)
        
        resultA = torch.cat([realA, fakeB, recA], 3).cpu().permute(0,2,3,1).data
        resultA = (resultA+1)/2
        resultB = torch.cat([realB, fakeA, recB], 3).cpu().permute(0,2,3,1).data
        resultB = (resultB+1)/2

        plt.figure(figsize=(8,8))
        for i,x in enumerate([resultA, resultB]):
            plt.subplot(2, 1, i+1)
            plt.imshow(x[0])
        plt.show()

    def G_backward(self, realA, realB, fakeA, fakeB):
        ## generate real A to fake B
        # fakeB = self.G_A(realA)
        D_A_result = self.D_A(fakeB)
        G_A_loss = self.MSE_loss(
            D_A_result,
            Variable(torch.ones(D_A_result.size())).type(self.dtype)
        )
        ## reconstruct fake B to A
        recA = self.G_B(fakeB)
        A_cycle_loss = self.L1_loss(recA, realA) * self.lambdaA
        ## generate real B to fake A
        # fakeA = self.G_A(realB)
        D_B_result = self.D_B(fakeA)
        G_B_loss = self.MSE_loss(
            D_B_result,
            Variable(torch.ones(D_B_result.size())).type(self.dtype)
        )
        ## reconstruct fake B to A
        recB = self.G_A(fakeA)
        B_cycle_loss = self.L1_loss(recB, realB) * self.lambdaB
        # Sum losses
        G_loss = G_A_loss + G_B_loss + A_cycle_loss + B_cycle_loss
        self.G_optimizer.zero_grad()
        G_loss.backward()
        self.G_optimizer.step()

        return (G_A_loss.cpu().data, 
            G_B_loss.cpu().data, 
            A_cycle_loss.cpu().data, 
            B_cycle_loss.cpu().data)

    def D_A_backward(self, realB, fakeB):
        # Train Discriminator A
        D_A_real = self.D_A(realB)
        D_A_real_loss = self.MSE_loss(
            D_A_real,
            Variable(torch.ones(D_A_real.size()).type(self.dtype))
        )
        fakeB = self.fakeB_store.query(fakeB)
        D_A_fake = self.D_A(fakeB)
        D_A_fake_loss = self.MSE_loss(
            D_A_fake,
            Variable(torch.zeros(D_A_fake.size()).type(self.dtype))
        )
        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        
        self.D_A_optimizer.zero_grad()
        D_A_loss.backward()
        self.D_A_optimizer.step()

        return D_A_loss.cpu().data

    def D_B_backward(self, realA, fakeA):
        # Train discriminator B
        D_B_real = self.D_B(realA)
        D_B_real_loss = self.MSE_loss(
            D_B_real,
            Variable(torch.ones(D_B_real.size()).type(self.dtype))
        )
        fakeA = self.fakeA_store.query(fakeA)
        D_B_fake = self.D_B(fakeA)
        D_B_fake_loss = self.MSE_loss(
            D_B_fake,
            Variable(torch.zeros(D_B_fake.size()).type(self.dtype))
        )
        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        
        self.D_B_optimizer.zero_grad()
        D_B_loss.backward()
        self.D_B_optimizer.step()

        return D_B_loss.cpu().data

    def train(self, num_epochs, data_A_loader, data_B_loader, print_every, num_img):
        for epoch in range(num_epochs):
            tic = time.time()
            print('Starting %d/%d' % (epoch+1, num_epochs))
            
            self.G_A.train()
            self.G_B.train()
            self.D_A.train()
            self.D_B.train()
            
            total_G_A_loss = 0
            total_G_B_loss = 0
            total_A_cycle_loss = 0
            total_B_cycle_loss = 0
            total_D_A_loss = 0
            total_D_B_loss = 0
            for (realA, _), (realB, _) in zip(data_A_loader, data_B_loader):
                realA = Variable(realA.type(self.dtype))
                realB = Variable(realB.type(self.dtype))
                fakeA = self.G_A(realB)
                fakeB = self.G_A(realA)

                G_A_loss, G_B_loss, A_cycle_loss, B_cycle_loss = self.G_backward(realA, realB, fakeA, fakeB)
                
                total_G_A_loss += G_A_loss
                total_G_B_loss += G_B_loss
                total_A_cycle_loss += A_cycle_loss
                total_B_cycle_loss += B_cycle_loss

                D_A_loss = self.D_A_backward(realA, fakeA)
                total_D_A_loss += D_A_loss
                
                D_B_loss = self.D_B_backward(realB, fakeB)
                total_D_B_loss += D_B_loss
                
            toc = time.time()
            self.train_hist[epoch]={
                'D_A_loss': total_D_A_loss / len(data_A_loader),
                'D_B_loss': total_D_B_loss / len(data_A_loader),
                'G_A_loss': total_G_A_loss / len(data_A_loader),
                'G_B_loss': total_G_B_loss / len(data_A_loader),
                'G_B_loss': total_G_B_loss / len(data_A_loader),
                'A_cycle_loss': total_A_cycle_loss / len(data_A_loader),
                'B_cycle_loss': total_B_cycle_loss / len(data_A_loader),
                'epoch_time': toc-tic
            }
            if (epoch+1) % print_every == 0:
                print(self.train_hist[epoch])
                self.show_img(realA[:1], realB[:1])