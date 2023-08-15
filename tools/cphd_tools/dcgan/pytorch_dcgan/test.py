import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import sys
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

import utils
from dcgan import Generator

def main():
    # define arguments
    parser = argparse.ArgumentParser(description='PyTorch GAN')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='enables cpu testing (no gpu)')
    parser.add_argument('--img_chan', type=int, default=1, help='image channel')
    parser.add_argument('--img_count', type=int, default=50,
                        help='number of images to create')
    parser.add_argument('--img_dim', type=int, default=64,
                        help='image dimension')
    parser.add_argument('--net_g', type=str, default='netG_4.pth',
                        help='generator network name to load (default: netG_4.pth)')
    parser.add_argument('--project_name', type=str, default='mnist',
                        help='project name, used to create output folder')
    parser.add_argument('--project_path', type=str, default=None,
                        help='project path')
    args = parser.parse_args()

    # ==================================================================================================================
    #                                       CHANGE HYPERPARAMETERS HERE
    # ==================================================================================================================

    # number of images in one batch, adjust this value according to your GPU memory
    batch_size = args.batch_size
    # True for GPU training, False for CPU training
    if args.cpu is False and torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    # number of channels, 1 for grayscale, 3 for rgb image
    image_channel = args.img_chan
    # image dimension
    image_dim = args.img_dim
    # number of iterations to create images
    image_count = args.img_count
    #  generator network name
    net_g = args.net_g
    # path to project
    if args.project_path is None:
        project_path = Path.cwd()
    else:
        project_path = Path(args.project_path)
    # project name
    project_name = args.project_name
    # ==========================================================================
    z_dim = 100
    g_hidden = image_dim
    # Change to None to get different results at each run
    seed = None
    # 0: random; 1: interpolation; 2: semantic calculation
    viz_mode = 0
    # ==========================================================================

    # path to input data and generator network name
    in_path = project_path.joinpath('output', project_name, 'train', net_g)
    # path to store output
    out_path = project_path.joinpath('output', project_name, 'test')
    # create log file and write outputs
    LOG_FILE = out_path.joinpath('log.txt')
    utils.clear_folder(out_path)
    print("Logging to {}\n".format(LOG_FILE))
    sys.stdout = utils.StdOut(LOG_FILE)
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

    # print arguments
    print("Project path: {}".format(project_path))
    print("Number of images to create: {}".format(image_count))
    print("Batch size: {}".format(batch_size))
    print("Image dimension: {}".format(image_dim))
    print("Image channel: {}".format(image_channel))
    print('Output path: {}\n'.format(out_path))

    # init GPU or CPU
    device = torch.device("cuda:0" if cuda else "cpu")

    # Generator
    netG = Generator(image_channel, z_dim, g_hidden)
    netG.load_state_dict(torch.load(in_path))
    netG.to(device)

    for i in range(image_count):
        if viz_mode == 0:
            viz_tensor = torch.randn(batch_size, z_dim, 1, 1, device=device)
        elif viz_mode == 1:
            load_vector = np.loadtxt('vec_20190317-223131.txt')
            xp = [0, 1]
            yp = np.vstack([load_vector[2], load_vector[9]])   # choose two exemplar vectors
            xvals = np.linspace(0, 1, num=batch_size)
            sample = interp1d(xp, yp, axis=0)
            viz_tensor = torch.tensor(sample(xvals).reshape(batch_size, z_dim, 1, 1), dtype=torch.float32, device=device)
        elif viz_mode == 2:
            load_vector = np.loadtxt('vec_20190317-223131.txt')
            z1 = (load_vector[0] + load_vector[6] + load_vector[8]) / 3.
            z2 = (load_vector[1] + load_vector[2] + load_vector[4]) / 3.
            z3 = (load_vector[3] + load_vector[4] + load_vector[6]) / 3.
            z_new = z1 - z2 + z3
            sample = np.zeros(shape=(batch_size, z_dim))
            for i in range(batch_size):
                sample[i] = z_new + 0.1 * np.random.normal(-1.0, 1.0, 100)
            viz_tensor = torch.tensor(sample.reshape(batch_size, z_dim, 1, 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            viz_sample = netG(viz_tensor)
            # viz_vector = utils.to_np(viz_tensor).reshape(batch_size, z_dim)
            # cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            # np.savetxt(out_path.joinpath('vec_{}.txt'.format(i)), viz_vector)
            vutils.save_image(viz_sample, out_path.joinpath('img_{}.png'.format(i)), nrow=10, normalize=True)

if __name__ == "__main__":
    main()