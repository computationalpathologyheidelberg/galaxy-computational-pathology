import argparse
import numpy as np
from pathlib import Path
import sys
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils
from dcgan import Generator, Discriminator, weights_init

def main():
    # define arguments
    parser = argparse.ArgumentParser(description='PyTorch GAN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='enables cpu training (no gpu)')
    parser.add_argument('--data_path', type=str, default='',
                        help='data path, if not given, MNISt will be downloaded')
    parser.add_argument('--epoch_num', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--img_chan', type=int, default=1, help='image channel')
    parser.add_argument('--img_dim', type=int, default=64,
                        help='image dimension')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--project_name', type=str, default='mnist',
                        help='project name, used to create output folder')
    parser.add_argument('--project_path', type=str, default=None,
                        help='project path')
    args = parser.parse_args()

    # number of images in one batch, adjust this value according to your GPU memory
    batch_size = args.batch_size
    # True for GPU training, False for CPU training
    if args.cpu is False and torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    # path to input data
    data_path = args.data_path
    # number if epochs for training (increase value for better results)
    epoch_num = args.epoch_num
    # number of channels, 1 for grayscale, 3 for rgb image
    image_channel = args.img_chan
    # image dimension
    image_dim = args.img_dim
    # learning rate (increase value for better results)
    lr = args.lr
    # project name
    project_name = args.project_name
    # path to project
    if args.project_path is None:
        project_path = Path.cwd()
    else:
        project_path = Path(args.project_path)
    # ==========================================================================
    # noise dimension
    z_dim = 100
    # number of generator filters
    g_hidden = image_dim
    # number of discriminator filters
    d_hidden = g_hidden
    # labels for classification (1=real, 0=fake)
    real_label = 1
    fake_label = 0
    # Change to None to get different results at each run
    seed = 1
    # path to store output files
    out_path = project_path.joinpath('output', project_name, 'train')
    # create log file and write outputs
    log_file = out_path.joinpath('log.txt')
    # ==========================================================================
    utils.clear_folder(out_path)
    print("Logging to {}\n".format(log_file))
    sys.stdout = utils.StdOut(log_file)
    print("PyTorch version: {}".format(torch.__version__))
    if cuda:
        print("cuda version: {}\n".format(torch.version.cuda))

    if seed is None:
        seed = np.random.randint(1, 10000)
    print("Random Seed: {}\n".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True      # May train faster but cost more memory

    # training parameters
    print("Learning rate: {}".format(lr))
    print("Batch size: {}".format(batch_size))
    print("Epochs: {}\n".format(epoch_num))

    # load dataset
    try:
        dataset = dset.ImageFolder(root=data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_dim),
                                       transforms.CenterCrop(image_dim),
                                       # transforms.RandomHorizontalFlip(),
                                       # transforms.RandomVerticalFlip(),
                                       # transforms.RandomRotation(180),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ]))
    except FileNotFoundError:
        print("no data available, using MNIST dataset instead")
        image_channel = 1
        data_path = project_path.joinpath('input/mnist/mnist')
        dataset = dset.MNIST(root=data_path, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(image_dim),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))
                             ]))

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    print("Data path: {}".format(data_path))
    print("Number of training images: {}".format(len(dataset)))
    print("Image dimension: {}".format(image_dim))
    print("Image channel: {}".format(image_channel))
    print('Output path: {}\n'.format(out_path))

    # init GPU or CPU
    device = torch.device("cuda:0" if cuda else "cpu")

    # Generator
    netG = Generator(image_channel, z_dim, g_hidden).to(device)
    netG.apply(weights_init)
    print(netG)

    # Discriminator
    netD = Discriminator(image_channel, d_hidden).to(device)
    netD.apply(weights_init)
    print(netD)

    # loss function
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    viz_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)

    # optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epoch_num):
        for i, data in enumerate(dataloader):
            x_real = data[0].to(device)
            real_label_t = torch.full((x_real.size(0),), real_label, dtype=torch.float32, device=device)
            fake_label_t = torch.full((x_real.size(0),), fake_label, dtype=torch.float32, device=device)

            # Update D with real data
            netD.zero_grad()
            y_real = netD(x_real)
            loss_D_real = criterion(y_real, real_label_t)
            loss_D_real.backward()

            # Update D with fake data
            z_noise = torch.randn(x_real.size(0), z_dim, 1, 1, device=device)
            x_fake = netG(z_noise)
            y_fake = netD(x_fake.detach())
            loss_D_fake = criterion(y_fake, fake_label_t)
            loss_D_fake.backward()
            optimizerD.step()

            # Update G with fake data
            netG.zero_grad()
            y_fake_r = netD(x_fake)
            loss_G = criterion(y_fake_r, real_label_t)
            loss_G.backward()
            optimizerG.step()

            if i % 100 == 0:
                print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                    epoch, i, len(dataloader),
                    loss_D_real.mean().item(),
                    loss_D_fake.mean().item(),
                    loss_G.mean().item()
                ))
                vutils.save_image(x_real, out_path.joinpath('real_samples.png'), normalize=True)
                with torch.no_grad():
                    viz_sample = netG(viz_noise)
                    vutils.save_image(viz_sample, out_path.joinpath('fake_samples_{}.png'.format(epoch)), normalize=True)
        torch.save(netG.state_dict(), out_path.joinpath('netG_{}.pth'.format(epoch)))
        torch.save(netD.state_dict(), out_path.joinpath('netD_{}.pth'.format(epoch)))

if __name__ == "__main__":
    main()