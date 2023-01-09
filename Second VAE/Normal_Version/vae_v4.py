from typing import Tuple
import math
import numpy
import torch
import torchvision

class Vae_v4(torch.nn.Module):

    def __init__(self,
                 input_d: Tuple[int],
                 n_frames: int,
                 n_latent: int,
                 batch: int):
        super(Vae_v4, self).__init__()
        self.n_frames = n_frames
        self.n_latent = n_latent
        self.input_d = input_d
        self.batch = batch

        x, y = input_d
        for i in range(4):
            x = math.floor((x - 1) / 3 + 1)
            y = math.floor((y - 1) / 3 + 1)
        self.hidden_x = x
        self.hidden_y = y

        self.enc_conv1 = torch.nn.Conv2d(self.n_frames, 64, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn1 = torch.nn.BatchNorm2d(64)
        self.enc_af1 = torch.nn.Hardtanh(inplace=True)
        
        self.enc_conv2 = torch.nn.Conv2d(64, 128, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn2 = torch.nn.BatchNorm2d(128)
        self.enc_af2 = torch.nn.Hardtanh(inplace=True)

        self.enc_conv3 = torch.nn.Conv2d(128, 256, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn3 = torch.nn.BatchNorm2d(256)
        self.enc_af3 = torch.nn.Hardtanh(inplace=True)

        self.enc_conv4 = torch.nn.Conv2d(256, 512, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn4 = torch.nn.BatchNorm2d(512)
        self.enc_af4 = torch.nn.Hardtanh(inplace=True)

        self.linear_mu = torch.nn.Linear(512 * self.hidden_x * self.hidden_y, self.n_latent)
        self.linear_var = torch.nn.Linear(512 * self.hidden_x * self.hidden_y, self.n_latent)

        self.dec_linear = torch.nn.Linear(self.n_latent, 512 * self.hidden_x * self.hidden_y)
        self.dec_linear_af = torch.nn.Hardtanh(inplace=True)

        self.dec_conv4 = torch.nn.ConvTranspose2d(512, 256, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(1, 2))
        self.dec_bn4 = torch.nn.BatchNorm2d(256)
        self.dec_af4 = torch.nn.Hardtanh(inplace=True)

        self.dec_conv3 = torch.nn.ConvTranspose2d(256, 128, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(1, 2))
        self.dec_bn3 = torch.nn.BatchNorm2d(128)
        self.dec_af3 = torch.nn.Hardtanh(inplace=True)

        self.dec_conv2 = torch.nn.ConvTranspose2d(128, 64, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(0, 2))
        self.dec_bn2 = torch.nn.BatchNorm2d(64)
        self.dec_af2 = torch.nn.Hardtanh(inplace=True)

        self.dec_conv1 = torch.nn.ConvTranspose2d(64, self.n_frames, (5, 5), stride=(3, 3), padding=(2, 2), output_padding=(2,0))
        self.dec_bn1 = torch.nn.BatchNorm2d(self.n_frames)
        self.dec_af1 = torch.nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_af1(x)
        
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.enc_af2(x)
        
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = self.enc_af3(x)
       
        x = self.enc_conv4(x)
        x = self.enc_bn4(x)
        x = self.enc_af4(x)
        
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x)
        return mu, var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec_linear(z)
        z = torch.reshape(z, [self.batch, 512, self.hidden_x, self.hidden_y])
        z = self.dec_conv4(z)
        z = self.dec_bn4(z)
        z = self.dec_af4(z)
       
        z = self.dec_conv3(z)
        z = self.dec_bn3(z)
        z = self.dec_af3(z)
        
        z = self.dec_conv2(z)
        z = self.dec_bn2(z)
        z = self.dec_af2(z)
        
        z = self.dec_conv1(z)
        z = self.dec_bn1(z)
        return self.dec_af1(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        out = self.decode(z)
        return out, mu, logvar

    def train_self(self,
                   train_path: str,
                   val_path: str,
                   weights: str,
                   epochs: int,
                   use_flows: bool = False) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        model = self.to(device)
        
        def npy_loader(path: str) -> torch.Tensor:
            sample = torch.from_numpy(numpy.load(path))
            sample = torch.swapaxes(sample, 1, 2)
            sample = torch.swapaxes(sample, 0, 1)
            sample = sample.nan_to_num(0)
            sample = ((sample + 64) / 128).clamp(0, 1)
            return sample.type(torch.FloatTensor) 

        if use_flows:
            train_set = torchvision.datasets.DatasetFolder(
                root=train_path,
                loader=npy_loader,
                extensions=['.npy'])
            val_set = torchvision.datasets.DatasetFolder(
                root=val_path,
                loader=npy_loader,
                extensions=['.npy'])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.input_d),
                torchvision.transforms.Grayscale()])
            train_set = torchvision.datasets.ImageFolder(
                root=train_path,
                transform=transforms)
            val_set = torchvision.datasets.ImageFolder(
                root=val_path,
                transform=transforms)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)
        
        optimizer = torch.optim.Adam(model.parameters())
        training_losses = []
        validation_losses = []
        for epoch in range(epochs):
            print('----------------------------------------------------')
            print(f'Epoch: {epoch}')

            model.train()
            epoch_tl = 0
            train_count = 0
            for data in train_loader:
                x, _ = data
               
                #print(f'MAX: {x.max()}')
                #print(f'MIN: {x.min()}')

                x = x.to(device)
                x_hat, mu, logvar = model(x)

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
                loss = mse_loss + kl_loss
                #ce_loss = torch.nn.functional.binary_cross_entropy(
                #    input=x_hat,
                #    target=x,
                #    reduction='sum')
                #loss = ce_loss + kl_loss
                epoch_tl += loss
                train_count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            training_losses.append(epoch_tl/train_count)
            print(f'Training Loss: {epoch_tl/train_count}')
            
            with torch.no_grad():
                epoch_vl = 0
                val_count = 0
                for data in val_loader:
                    x, _ = data
                    x = x.to(device)
                    x_hat, mu, logvar = model(x)

                    kl_loss = torch.mul(
                        input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                        other=0.5)
                    mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
                    epoch_vl += mse_loss + kl_loss
                    #ce_loss = torch.nn.functional.binary_cross_entropy(
                    #    input=x_hat,
                    #    target=x,
                    #    reduction='sum')
                    #epoch_vl += ce_loss + kl_loss
                    val_count += 1
                validation_losses.append(epoch_vl/val_count)
                print(f'Validation Loss: {epoch_vl/val_count}')
    
            print('----------------------------------------------------')
        print('Training finished, saving weights...')
        with open('/content/Losses/training_losses.txt', 'w') as f:
            for line in training_losses:
                f.write(f"{line}\n")
        with open('/content/Losses/validation_losses.txt', 'w') as f:
            for line in validation_losses:
                f.write(f"{line}\n")
        torch.save(model, weights)