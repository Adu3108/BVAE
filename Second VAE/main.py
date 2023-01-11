import argparse
import json
import numpy
import torch
import torchvision
import os
import matplotlib.pyplot as plt
from Binarized_Version.binary_vae_v1 import Binary_Vae_v1
from Normal_Version.vae_v1 import Vae_v1
from Normal_Version.vae_v2 import Vae_v2
from Normal_Version.vae_v3 import Vae_v3
from Normal_Version.vae_v4 import Vae_v4
from Normal_Version.vae_v5 import Vae_v5

parser = argparse.ArgumentParser('Train or convert a VAE model.')
parser.add_argument(
    'action',
    choices=['train', 'calibrate'],
    metavar='ACTION')
parser.add_argument(
    '--version',
    required = True,
    help='The version of the network to be used')
parser.add_argument(
    '--type',
    choices=['normal','binarized'],
    required = True,
    help='Type of network to be used')
parser.add_argument(
    '--weights',
    help='Path to weights file')
parser.add_argument(
    '--n_latent',
    help='Number of latent variables in the model.')
parser.add_argument(
    '--dimensions',
    help='Dimensions of input image accepted by the network (height x width).')
parser.add_argument(
    '--train_set',
    help='Path to the training set.')
parser.add_argument(
    '--validation_set',
    help='Path to the cross validation set (for monitoring training only, does not affect weight calculation).')
parser.add_argument(
    '--cal_set',
    help='Path to calibration set (calibrate action only).')
parser.add_argument(
    '--batch',
    help='Batch size to use for training.')
parser.add_argument(
    '--epochs',
    help='Epochs to train.')
parser.add_argument(
    '--flows',
    type=int,
    default=0,
    help='Training set is optical flows (.npy cubic tensors).')
args = parser.parse_args()

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)

Final_VAE = None

if args.version == '1':
    if args.type == "binarized":
        Final_VAE = Binary_Vae_v1
    else:
        Final_VAE = Vae_v1
elif args.version == '2':
    Final_VAE = Vae_v2
elif args.version == '3':
    Final_VAE = Vae_v3
elif args.version == '4':
    Final_VAE = Vae_v4
elif args.version == '5':
    Final_VAE = Vae_v5

if args.action == 'train':
    if not os.path.exists("/content/Losses"):
        os.mkdir("/content/Losses")
    model = Final_VAE(
        input_d=tuple([int(i) for i in args.dimensions.split('x')]),
        n_frames=1 if args.flows <= 0 else args.flows,
        n_latent=int(args.n_latent),
        batch=int(args.batch))
    model.train_self(
        train_path=args.train_set,
        val_path=args.validation_set,
        weights=args.weights,
        epochs=int(args.epochs),
        use_flows=False if args.flows <= 0 else True)

if args.action == 'calibrate':
    model = torch.load(args.weights)
    model.eval()

    def npy_loader(path: str) -> torch.Tensor:
        sample = torch.from_numpy(numpy.load(path))
        sample = torch.swapaxes(sample, 1, 2)
        sample = torch.swapaxes(sample, 0, 1)
        sample = sample.nan_to_num(0)
        sample = ((sample + 64) / 128).clamp(0, 1)
        return sample.type(torch.FloatTensor)

    if args.flows <= 0:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(model.input_d),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()])
        cal_set = torchvision.datasets.ImageFolder(
            root=args.cal_set,
            transform=transforms)
    else:
        cal_set = torchvision.datasets.DatasetFolder(
            root=args.cal_set,
            loader=npy_loader,
            extensions=['.npy'])
    cal_loader = torch.utils.data.DataLoader(
        dataset=cal_set,
        batch_size=model.batch,
        shuffle=True,
        drop_last=True)
    
    kl_losses = []
    #ce_losses = []
    mse_losses = []
    final = []
    index = 0
    class_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for data,idx in cal_loader:
            x = data
            class_list.append(int(idx))
            final_input = x
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            
            kl_loss = torch.mul(
                input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, 1),
                other=0.5)
            #ce_loss = -torch.sum(
            #        x_hat * x.log2().nan_to_num(-100) + (1 - x_hat) * (1 - x).log2().nan_to_num(-100),
            #        (1, 2, 3))
            mse_loss = torch.sum((x - x_hat).pow(2), (1, 2, 3))
            l = [index, final_input, x_hat, kl_loss.detach().cpu().numpy(), mse_loss.detach().cpu().numpy()]
            final.append(l)
            kl_losses.extend(list(kl_loss.detach().cpu().numpy()))
            #ce_losses.extend(list(ce_loss.detach().cpu().numpy()))
            mse_losses.extend(list(mse_loss.detach().cpu().numpy()))
            index += 1
    kl_losses_2 = kl_losses[:]
    mse_losses_2 = mse_losses[:]
    kl_losses.sort()
    mse_losses.sort()
    #ce_losses.sort()
    kl_losses = [i.item() for i in kl_losses]
    mse_losses = [i.item() for i in mse_losses]
    #ce_losses = [i.item() for i in ce_losses]
    # with open(f'cal_{".".join(args.weights.split(".")[:-1])}.json', 'w') as cal_f:
    #     cal_f.write(json.dumps({'kl_loss': kl_losses, 'mse_loss': mse_losses}))
    for i in range(len(final)):
        input_matrix = final[i][1][0][0] 
        output_matrix = final[i][2][0][0] 
        matrix_shape = input_matrix.shape
        input_img = input_matrix.cpu().numpy()
        output_img = output_matrix.cpu().numpy()
        numpy.savetxt('/content/Results/input.csv', input_img, delimiter=',')
        numpy.savetxt('/content/Results/output.csv', output_img, delimiter=',')
        plt.imsave(f'/content/Results/Input/{i}_{class_list[i]}.png', input_img, cmap='gray')
        plt.imsave(f'/content/Results/Reconstructed/{i}_{class_list[i]}.png', output_img, cmap='gray')
    final_kl_dict = {}
    final_mse_dict = {}
    for i in final:
        final_kl_dict[f'{i[0]}'] = str(i[3])
        final_mse_dict[f'{i[0]}'] = str(i[4])
    with open("/content/Losses/kl_loss.json", "w") as KL_Json:
        json.dump(final_kl_dict, KL_Json)
    with open("/content/Losses/mse_loss.json","w") as MSE_Json:
        json.dump(final_mse_dict, MSE_Json)