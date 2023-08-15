# DCGAN in PyTorch for Histopathological Image Generation
An implementation of a DCGAN (Deep Convolutional Generative Adversarial Network) for the generation of histopathological images, i.e classified glomerulus pathologies of the kidney.
The Code is derived from Chapter 4 & 5 from [Hany, J. & Walters, G. (2019). Hands-on Generative Adversarial Networks with PyTorch 1.0 : implement next-generation neural networks to build powerful gan models using python](https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x#hands-on-generative-adversarial-networks-with-pytorch-10).

## Prerequisites
- Linux OS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
    ```bash
    git clone https://github.com/m4ln/pytorch_dcgan.git
    cd pytorch_dcgan
    ```

- Install dependencies via [pip](https://pypi.org/project/pip/)
  ```
  pip install -r requirements.txt
  ```
  It might be necessary to install PyTorch manually from https://pytorch.org/get-started/locally/

### Training
- When not providing input arguments the model is trained on the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset
  ```bash
  python train.py
  ```
- To train on your own data provide the arguments via argparse (check inside `train.py`)
- The directory to your input data should contain a subfolder of images for each class
  
### Generating Fake Samples/Testing
- To generate new samples run (by default using MNIST trained model as in `train.py`)):
  ```bash
  python test.py
  ```
- To test on your own data provide the arguments via argparse (check inside `test.py`)