import argparse
import math
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from Binarized_Version.Binary_bvae_v1 import Binary_BetaVae_v1, Binary_Encoder_v1
from Binarized_Version.Binary_bvae_v2 import Binary_BetaVae_v2, Binary_Encoder_v2
from Binarized_Version.Binary_bvae_v3 import Binary_BetaVae_v3, Binary_Encoder_v3
from Binarized_Version.Binary_bvae_v4 import Binary_BetaVae_v4, Binary_Encoder_v4
from Normal_Version.bvae_v1 import BetaVae_v1, Encoder_v1
from Normal_Version.bvae_v2 import BetaVae_v2, Encoder_v2
from Normal_Version.bvae_v3 import BetaVae_v3, Encoder_v3
from Normal_Version.bvae_v4 import BetaVae_v4, Encoder_v4

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train or convert a BetaVAE model.')
    parser.add_argument(
        'action',
        choices=['train', 'convert','test'],
        metavar='ACTION',
        help='Train a new network or convert one to encoder-only.')
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
        help='Path to weights file (initial weights to use if the train '
             'option is selected).')
    parser.add_argument(
        '--beta',
        help='Beta value for training.')
    parser.add_argument(
        '--n_latent',
        help='Number of latent variables in the model')
    parser.add_argument(
        '--dimensions',
        help='Dimension of input image accepted by the network (height x '
             'width).')
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Network accepts grayscale images if this flag is selected.')
    parser.add_argument(
        '--dataset',
        help='Path to dataset.  This data set should be a folder containing '
             'subfolders for level of variation for each partition.')
    parser.add_argument(
        '--output_path',
        help='Path to directory where input images and their reconstructions will be stored')
    args = parser.parse_args()

    Final_VAE = None
    Final_Encoder = None

    if args.version == '1':
        if args.type == "binarized":
            Final_VAE = Binary_BetaVae_v1
            Final_Encoder = Binary_Encoder_v1
        else:
            Final_VAE = BetaVae_v1
            Final_Encoder = Encoder_v1
    elif args.version == '2':
        if args.type == "binarized":
            Final_VAE = Binary_BetaVae_v2
            Final_Encoder = Binary_Encoder_v2
        else:
            Final_VAE = BetaVae_v2
            Final_Encoder = Encoder_v2
    elif args.version == '3':
        if args.type == "binarized":
            Final_VAE = Binary_BetaVae_v3
            Final_Encoder = Binary_Encoder_v3
        else:
            Final_VAE = BetaVae_v3
            Final_Encoder = Encoder_v3
    elif args.version == '4':
        if args.type == "binarized":
            Final_VAE = Binary_BetaVae_v4
            Final_Encoder = Binary_Encoder_v4
        else:
            Final_VAE = BetaVae_v4
            Final_Encoder = Encoder_v4

    if args.action == 'train':
        if not (args.beta and args.n_latent and args.dimensions and args.dataset):
            print('The optional arguments "--beta", "--n_latent", "--dimensions", and "--dataset" are required for training')
            sys.exit(1)
        n_latent = int(args.n_latent)
        beta = float(args.beta)
        input_dimensions = tuple([int(i) for i in args.dimensions.split('x')])
        print(f'Starting training for input size {args.dimensions}')
        print(f'beta={beta}')
        print(f'n_latent={n_latent}')
        print(f'Using data set {args.dataset}')
        network = Final_VAE(
            n_latent,
            beta,
            n_chan=1 if args.grayscale else 3,
            input_d=input_dimensions)
        network.train_self(
            data_path=args.dataset,
            epochs=100,
            weights_file=f'bvae_n{n_latent}_b{beta}_'
                         f'{"bw" if args.grayscale else ""}_'
                         f'{"x".join([str(i) for i in input_dimensions])}.pt')

    elif args.action == 'convert':
        if not args.weights:
            print('The optional argument "--weights" is required for model conversion.')
            sys.exit(1)
        print(f'Converting model {args.weights} to encoder-state-dict-only '
              f'version...')
        full_model = torch.load(args.weights)
        encoder = Final_Encoder(
            n_latent=full_model.n_latent,
            n_chan=full_model.n_chan,
            input_d=full_model.input_d)
        full_dict = full_model.state_dict()
        encoder_dict = encoder.state_dict()
        for key in encoder_dict:
            encoder_dict[key] = full_dict[key]
        torch.save(encoder_dict, f'enc_only_{args.weights}')

    elif args.action == 'test':
        if not (args.beta and args.n_latent and args.dimensions and args.dataset and args.weights and args.output_path):
            print('The optional argument "--weights", "--beta", "--n_latent", "--dimensions", "--output_path" and "--dataset" is required for model conversion.')
            sys.exit(1)
        n_latent = int(args.n_latent)
        beta = float(args.beta)
        input_dimensions = tuple([int(i) for i in args.dimensions.split('x')])
        print(f'Starting training for input size {args.dimensions}')
        print(f'beta={beta}')
        print(f'n_latent={n_latent}')
        print(f'Using data set {args.dataset}')
        network = Final_VAE(
            n_latent,
            beta,
            n_chan=1 if args.grayscale else 3,
            input_d=input_dimensions)
        (input_final,output_final) = network.testing(args.dataset, args.weights)
        error_array = []
        for i in range(len(input_final)):
          input_matrix = input_final[i][0][0] 
          output_matrix = output_final[i][0][0] 
          matrix_shape = input_matrix.shape
          input_img = input_matrix.cpu().numpy()
          output_img = output_matrix.cpu().detach().numpy()
          plt.imsave(args.output_path + f'/Input/{i}.png', input_img, cmap='gray')
          plt.imsave(args.output_path + f'/Reconstructed/{i}.png', output_img, cmap='gray')
          error = 0
          for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
              error += (output_matrix[i][j] - input_matrix[i][j])**2
          error_array.append(math.sqrt(error))
        file1 = open(args.output_path + "/errors.txt", "w")
        for i in error_array:
          file1.write(str(i))
          file1.write("\n")
        file1.close()