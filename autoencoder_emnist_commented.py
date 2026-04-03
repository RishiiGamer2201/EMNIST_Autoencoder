# ============================================================
#  AUTOENCODER MODELS ON EMNIST — FULLY COMMENTED VERSION
#  Author  : Rishii Kumar Singh
#  Dataset : EMNIST Letters (26 handwritten letters A-Z)
#  Task    : Image Compression & Reconstruction
# ============================================================
#
#  WHAT IS AN AUTOENCODER?
#  An autoencoder is a neural network that learns to compress (encode) an image
#  into a small vector of numbers (the "latent space"), and then reconstruct
#  (decode) the image back from that small vector.
#
#  WHY IS THIS USEFUL?
#  - Image compression (storing images in less space)
#  - Denoising (removing noise from images)
#  - Anomaly detection (if reconstruction is bad, it's an anomaly)
#  - Learning meaningful representations of data
#
#  PIPELINE:
#  Input Image (784 pixels) → [Encoder] → Latent z (32 or 64 numbers)
#                                       → [Decoder] → Reconstructed Image (784 pixels)
#  Goal: make reconstructed image as close to original as possible
#
#  MODELS WE BUILD:
#  ANN-1 : Simple fully-connected (no convolutions), bottleneck=32
#  ANN-2 : Deeper fully-connected + BatchNorm + Dropout, bottleneck=64
#  CNN-1 : Lightweight convolutional, bottleneck=32
#  CNN-2 : Deep convolutional + BatchNorm, bottleneck=64
# ============================================================


# ── SECTION 0: IMPORTS ──────────────────────────────────────────────────────────
# These are libraries we need — think of them as toolboxes

import os           # for creating folders, handling file paths
import time         # for measuring how long training takes
import random       # for random number generation (used in reproducibility)
import numpy as np  # NumPy: for numerical arrays (the backbone of ML data handling)

import matplotlib           # the overall plotting library
import matplotlib.pyplot as plt  # pyplot: the specific module for making graphs and images

# scikit-learn: a machine learning library; we use it only for dimensionality reduction viz
from sklearn.decomposition import PCA   # PCA: reduces high-dimensional data to 2D for plotting
from sklearn.manifold import TSNE       # t-SNE: another way to visualise high-dimensional data in 2D

import torch                            # PyTorch: the main deep learning framework
import torch.nn as nn                   # nn: module containing all neural network building blocks (Linear, Conv2d, etc.)
import torch.optim as optim             # optim: contains optimisers like Adam which update model weights
from torch.utils.data import DataLoader, random_split  # DataLoader: batches and shuffles data; random_split: splits dataset
from torchvision import datasets, transforms           # torchvision: contains image datasets and image transforms


# ── SECTION 1: REPRODUCIBILITY & DEVICE ────────────────────────────────────────
# "Reproducibility" means: if you run this code twice, you get the same results

SEED = 42                        # a fixed random number — any number works, 42 is convention
random.seed(SEED)                # tell Python's random module to use this seed
np.random.seed(SEED)             # tell NumPy to use this seed
torch.manual_seed(SEED)          # tell PyTorch to use this seed for CPU operations


def get_device():
    # This function figures out whether to use GPU (fast) or CPU (slow but always works)
    # GPU = Graphics Processing Unit: originally for games, now used heavily for AI because
    #       it can do thousands of math operations in parallel
    # CPU = Central Processing Unit: the main processor in your computer

    if not torch.cuda.is_available():
        # torch.cuda.is_available() returns True if a compatible NVIDIA GPU is found
        return torch.device("cpu")    # no GPU found, use CPU

    try:
        major, minor = torch.cuda.get_device_capability()
        # get_device_capability() returns the "compute capability" of the GPU
        # e.g., (7, 5) for a T4, (6, 0) for a P100
        # PyTorch 2.x dropped support for sm_60 (Compute Capability 6.0 = P100)
        # Only sm_70 and above (Volta, Turing, Ampere) work with PyTorch 2.x

        if major >= 7:
            # major version 7 or above = compatible (T4=7.5, V100=7.0, A100=8.0)
            return torch.device("cuda")    # use GPU
        else:
            # Kaggle's P100 is sm_60 — not compatible with modern PyTorch
            print(f"⚠️  GPU {torch.cuda.get_device_name()} (sm_{major}{minor}) "
                  f"not compatible with PyTorch {torch.__version__} (needs sm_70+)")
            print("   → Falling back to CPU.")
            return torch.device("cpu")    # fall back to CPU
    except Exception as e:
        # if anything goes wrong probing the GPU, just use CPU safely
        print(f"⚠️  GPU probe failed: {e}. Using CPU.")
        return torch.device("cpu")


DEVICE  = get_device()           # DEVICE is either "cuda" (GPU) or "cpu"

OUT_DIR = "/kaggle/working/outputs"   # folder where we save all plots and model weights
os.makedirs(OUT_DIR, exist_ok=True)   # create the folder if it doesn't exist yet
                                      # exist_ok=True means no error if folder already exists

print(f"Device  : {DEVICE}")          # show which device we're using
print(f"PyTorch : {torch.__version__}")  # show PyTorch version
print(f"Outputs : {OUT_DIR}")         # show where outputs will be saved


# ── SECTION 2: HYPERPARAMETERS ──────────────────────────────────────────────────
# Hyperparameters are settings you choose BEFORE training (not learned by the model)

IMG_SIZE   = 28     # EMNIST images are 28×28 pixels
BATCH_SIZE = 256    # how many images to process at once (larger = faster but needs more RAM)
VAL_FRAC   = 0.10   # 10% of training data used for validation (to check generalisation)
EPOCHS     = 40     # how many times to loop through the full training dataset
LR         = 1e-3   # learning rate: how big each weight update step is (1e-3 = 0.001)


# ── SECTION 3: DATA LOADING ─────────────────────────────────────────────────────
# We use the EMNIST "letters" split: 26 classes (A-Z), 28×28 greyscale images
# Training: 124,800 images → split into 112,320 train + 12,480 validation
# Test    :  20,800 images (never seen during training, used only for final evaluation)

transform = transforms.Compose([
    transforms.ToTensor()   # convert PIL image (0-255 integer) to PyTorch tensor (0.0-1.0 float)
                            # Compose means "apply these transforms in sequence"
])

# Download the EMNIST Letters dataset (downloads ~530MB the first time, cached after that)
full_train = datasets.EMNIST(
    root="/kaggle/working/data",   # where to store downloaded data
    split="letters",               # which split to use: "letters" = 26 letter classes
    train=True,                    # True = training set, False = test set
    download=True,                 # download if not already present
    transform=transform            # apply our transform (ToTensor) to each image
)

test_set = datasets.EMNIST(
    root="/kaggle/working/data",
    split="letters",
    train=False,                   # this is the test set
    download=True,
    transform=transform
)

# Split the training set into train (90%) and validation (10%)
n_val = int(len(full_train) * VAL_FRAC)     # number of validation images (12,480)
n_trn = len(full_train) - n_val             # number of training images (112,320)
train_set, val_set = random_split(
    full_train,                             # dataset to split
    [n_trn, n_val],                         # sizes of each split
    generator=torch.Generator().manual_seed(SEED)  # fixed seed so split is always the same
)

# DataLoader wraps a dataset and serves it in batches
# num_workers=2 means 2 background CPU threads pre-load the next batch while training
# pin_memory=True speeds up GPU transfers by keeping data in "pinned" (non-pageable) RAM
kw = dict(
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=(DEVICE.type == "cuda")   # only pin memory if we're actually using GPU
)
train_loader = DataLoader(train_set, shuffle=True,  **kw)  # shuffle=True randomises order each epoch
val_loader   = DataLoader(val_set,   shuffle=False, **kw)  # shuffle=False: consistent validation
test_loader  = DataLoader(test_set,  shuffle=False, **kw)

print(f"Train : {len(train_set):>8,}")
print(f"Val   : {len(val_set):>8,}")
print(f"Test  : {len(test_set):>8,}")
print(f"Image size : {IMG_SIZE}×{IMG_SIZE}  |  Classes : 26 letters")

INPUT_DIM = IMG_SIZE * IMG_SIZE   # 28×28 = 784 pixels (used by ANN models which need flat input)


# ── SECTION 4: SAMPLE VISUALISATION ────────────────────────────────────────────
# Let's see what the data looks like before training anything

def visualise_samples(loader, save_path):
    imgs, labels = next(iter(loader))   # next(iter(loader)) grabs one batch (256 images)
                                         # imgs shape: (256, 1, 28, 28) — batch of greyscale images
                                         # labels shape: (256,) — class index for each image

    fig, axes = plt.subplots(2, 16, figsize=(20, 3))
    # plt.subplots(rows, cols) creates a grid of subplots
    # figsize=(width_inches, height_inches)

    for i in range(32):   # show first 32 images (2 rows × 16 columns)
        ax = axes[i // 16][i % 16]   # pick the right subplot (i//16 = row, i%16 = col)
        ax.imshow(imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        # squeeze() removes the channel dimension: (1,28,28) → (28,28) which imshow needs
        # cmap='gray' shows greyscale, vmin/vmax fix the colour scale to 0-1
        letter = chr(labels[i].item() + 64)
        # EMNIST labels: 1='a', 2='b', ..., 26='z'
        # chr(65)='A', so chr(1+64)='A', chr(2+64)='B', etc.
        # .item() converts a 1-element tensor to a plain Python int
        ax.set_title(letter, fontsize=8)   # show letter above image
        ax.axis('off')                     # hide axes/ticks (cleaner look)

    plt.suptitle("EMNIST Letters — Test Set Samples", fontweight='bold')
    plt.tight_layout()                     # auto-adjust spacing between subplots
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # dpi=150: resolution (dots per inch); bbox_inches='tight': crop whitespace
    plt.show()

visualise_samples(test_loader, f"{OUT_DIR}/0_dataset_samples.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════
#
#  In PyTorch, every model is a class that inherits from nn.Module.
#  You must define two methods:
#    __init__  : define all layers (called once when you create the model)
#    forward   : define how data flows through layers (called every time you run the model)
#
#  KEY LAYERS USED:
#  nn.Linear(in, out)        : Fully connected layer — connects every input neuron to every output neuron
#  nn.Conv2d(in_ch, out_ch, k, stride, padding) : Convolutional layer — applies a sliding filter
#  nn.ConvTranspose2d(...)   : Transposed conv (aka "deconv") — upsamples spatial dimensions
#  nn.BatchNorm1d/2d(n)      : Batch Normalisation — normalises activations, speeds up training
#  nn.ReLU()                 : Activation: max(0, x) — introduces non-linearity
#  nn.LeakyReLU(slope)       : Like ReLU but allows small negative values (avoids "dead neurons")
#  nn.Dropout(p)             : Randomly zeros p% of neurons during training — prevents overfitting
#  nn.Sigmoid()              : Output in range (0,1) — needed since pixel values are in [0,1]
#  nn.Sequential(...)        : Chains layers in order, passes output of one as input to next


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 1: ANN-1 — Shallow Fully-Connected Autoencoder (Bottleneck = 32)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Architecture:
#  ENCODER: 784 → Linear(256) → ReLU → Linear(128) → ReLU → Linear(32) → ReLU
#  DECODER:  32 → Linear(128) → ReLU → Linear(256) → ReLU → Linear(784) → Sigmoid
#
#  The image is first FLATTENED from (1,28,28) to a vector of 784 numbers
#  The encoder progressively shrinks: 784 → 256 → 128 → 32
#  The decoder mirrors this in reverse: 32 → 128 → 256 → 784
#  Then the 784 vector is reshaped back to (1,28,28)

class ANN_AE_1(nn.Module):
    """Shallow FC Autoencoder | bottleneck=32 | Simple baseline"""

    def __init__(self, bottleneck=32):
        # __init__ is called when you do: model = ANN_AE_1(bottleneck=32)
        super().__init__()              # must call parent class __init__ (nn.Module requirement)
        self.bottleneck = bottleneck    # store bottleneck size as attribute

        # ENCODER: compresses 784 → bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),  # Layer 1: 784 inputs → 256 outputs (learns 784×256 weights)
            nn.ReLU(),                  # Activation: ReLU(x) = max(0,x) — non-linearity
            nn.Linear(256, 128),        # Layer 2: 256 → 128
            nn.ReLU(),
            nn.Linear(128, bottleneck), # Layer 3: 128 → 32 (the bottleneck!)
            nn.ReLU(),
        )

        # DECODER: expands bottleneck → 784
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), # Layer 1: 32 → 128 (mirror of encoder)
            nn.ReLU(),
            nn.Linear(128, 256),        # Layer 2: 128 → 256
            nn.ReLU(),
            nn.Linear(256, INPUT_DIM),  # Layer 3: 256 → 784 (back to full image size)
            nn.Sigmoid(),               # Sigmoid squashes output to [0,1] = valid pixel range
        )

    def forward(self, x):
        # forward() is called when you do: output, z = model(input_image)
        # x shape coming in: (batch_size, 1, 28, 28) — a batch of images
        z   = self.encoder(x.view(x.size(0), -1))
        # x.view(batch_size, -1): flatten image to (batch_size, 784)
        # -1 means "figure out this dimension automatically" = 1×28×28 = 784
        # self.encoder(...): pass flattened image through encoder → z shape: (batch_size, 32)
        out = self.decoder(z).view(x.size(0), 1, IMG_SIZE, IMG_SIZE)
        # self.decoder(z): expand z back to 784-d vector
        # .view(..., 1, 28, 28): reshape flat vector back to image shape
        return out, z   # return both the reconstructed image AND the latent code z


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 2: ANN-2 — Deep Fully-Connected Autoencoder (Bottleneck = 64)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Architecture:
#  ENCODER: 784 → 512(BN+LReLU+Drop) → 256(BN+LReLU+Drop) → 128(BN+LReLU) → 64
#  DECODER:  64 → 128(BN+LReLU) → 256(BN+LReLU+Drop) → 512(BN+LReLU+Drop) → 784
#
#  Key differences from ANN-1:
#  1. Deeper: 4 layers instead of 3
#  2. Wider: starts at 512 instead of 256
#  3. BatchNorm: normalises activations → faster, more stable training
#  4. LeakyReLU: like ReLU but slope=0.1 for negatives (avoids "dying ReLU" problem)
#  5. Dropout(0.2): randomly zeroes 20% of neurons during training → reduces overfitting
#  6. Larger bottleneck: 64 instead of 32 → less compression but better reconstruction

class ANN_AE_2(nn.Module):
    """Deep FC Autoencoder | bottleneck=64 | BatchNorm + LeakyReLU + Dropout"""

    def __init__(self, bottleneck=64, drop=0.2):
        # drop = dropout probability (0.2 = zero out 20% of neurons randomly)
        super().__init__()
        self.bottleneck = bottleneck

        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),   # 784 → 512
            nn.BatchNorm1d(512),         # BN1d: normalise 512-d activations across the batch
                                         # makes each neuron's output have mean≈0, std≈1
            nn.LeakyReLU(0.1),           # LeakyReLU with negative slope = 0.1
            nn.Dropout(drop),            # randomly zero 20% of the 512 values during training
            nn.Linear(512, 256),         # 512 → 256
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(drop),
            nn.Linear(256, 128),         # 256 → 128
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, bottleneck),  # 128 → 64 (bottleneck, no activation — raw values)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),  # 64 → 128
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),         # 128 → 256
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(drop),
            nn.Linear(256, 512),         # 256 → 512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(drop),
            nn.Linear(512, INPUT_DIM),   # 512 → 784
            nn.Sigmoid(),                # output in [0,1]
        )

    def forward(self, x):
        z   = self.encoder(x.view(x.size(0), -1))   # flatten then encode → (B, 64)
        out = self.decoder(z).view(x.size(0), 1, IMG_SIZE, IMG_SIZE)  # decode then reshape
        return out, z


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 3: CNN-1 — Lightweight Convolutional Autoencoder (Bottleneck = 32)
# ─────────────────────────────────────────────────────────────────────────────
#
#  WHY CONVOLUTIONS INSTEAD OF FULLY-CONNECTED?
#  - A Conv2d layer learns a small "filter" (e.g. 3×3) and slides it across the image
#  - The same filter detects the same feature (edge, curve) anywhere in the image
#  - This is called "weight sharing" and is much more parameter-efficient than Linear
#  - CNNs respect spatial structure: nearby pixels are processed together
#
#  Architecture:
#  ENCODER:
#    Conv(1→8,  k=3, stride=2, pad=1) : (1,28,28)  → (8,14,14)   [spatial halved]
#    Conv(8→16, k=3, stride=2, pad=1) : (8,14,14)  → (16,7,7)    [spatial halved]
#    Conv(16→32,k=3, stride=2, pad=1) : (16,7,7)   → (32,4,4)    [spatial halved]
#    Flatten                           : (32,4,4)   → 512
#    Linear(512 → 32)                 : 512 → bottleneck
#
#  DECODER:
#    Linear(32 → 512)                 : bottleneck → 512
#    Reshape                          : 512 → (32,4,4)
#    ConvT(32→16,k=3,s=2,p=1,op=0)  : (32,4,4)   → (16,7,7)    [spatial doubled]
#    ConvT(16→8, k=3,s=2,p=1,op=1)  : (16,7,7)   → (8,14,14)   [spatial doubled]
#    ConvT(8→1,  k=3,s=2,p=1,op=1)  : (8,14,14)  → (1,28,28)   [spatial doubled]
#
#  stride=2 means "move the filter 2 pixels at a time" → halves spatial dimensions
#  ConvTranspose2d (deconv) does the reverse — upsamples spatial dimensions

class CNN_AE_1(nn.Module):
    """Lightweight CNN Autoencoder | bottleneck=32 | Only 45K parameters"""

    def __init__(self, bottleneck=32):
        super().__init__()
        self.bottleneck = bottleneck

        # ENCODER: a series of conv layers that shrink spatial dims while increasing channels
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # (B,1,28,28) → (B,8,14,14)
            # kernel_size=3: 3×3 filter
            # stride=2: move 2 pixels at a time → output is half the spatial size
            # padding=1: pad 1 pixel of zeros around image → keeps output size predictable
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # (B,8,14,14) → (B,16,7,7)
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),# (B,16,7,7)  → (B,32,4,4)
            nn.ReLU(),
        )

        # After conv: (B,32,4,4) → flatten to (B, 512) → linear to (B, 32)
        self.enc_fc = nn.Linear(32 * 4 * 4, bottleneck)   # 512 → 32

        # DECODER: first expand, then upsample with transposed convolutions
        self.dec_fc = nn.Linear(bottleneck, 32 * 4 * 4)   # 32 → 512

        self.decoder_conv = nn.Sequential(
            # ConvTranspose2d is the "inverse" of Conv2d — upsamples the spatial dimensions
            # output_padding: adds extra pixels to ensure output size exactly matches
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0), # → (B,16,7,7)
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # → (B,8,14,14)
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),   # → (B,1,28,28)
            nn.Sigmoid(),   # output pixel values in [0,1]
        )

    def forward(self, x):
        h  = self.encoder_conv(x)                       # run through conv layers
        z  = self.enc_fc(h.view(x.size(0), -1))         # flatten then linear → latent z
        h2 = self.dec_fc(z).view(-1, 32, 4, 4)          # expand z → reshape to (B,32,4,4)
        return self.decoder_conv(h2), z                  # run through deconv layers


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL 4: CNN-2 — Deep Convolutional Autoencoder (Bottleneck = 64)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Architecture (deeper than CNN-1):
#  ENCODER:
#    Conv(1→16,  k=3,s=1,p=1)+BN2d+LReLU : (1,28,28)  → (16,28,28)  [same size, more channels]
#    Conv(16→32, k=3,s=2,p=1)+BN2d+LReLU : (16,28,28) → (32,14,14)
#    Conv(32→64, k=3,s=2,p=1)+BN2d+LReLU : (32,14,14) → (64,7,7)
#    Conv(64→128,k=3,s=2,p=1)+BN2d+LReLU : (64,7,7)   → (128,4,4)
#    Dropout → Linear(2048 → 64)
#
#  DECODER:
#    Linear(64 → 2048)+LReLU+Dropout → Reshape(128,4,4)
#    ConvT(128→64)+BN2d+LReLU  → (64,7,7)
#    ConvT(64→32)+BN2d+LReLU   → (32,14,14)
#    ConvT(32→16)+BN2d+LReLU   → (16,28,28)
#    Conv(16→1)+Sigmoid         → (1,28,28)  [regular conv to avoid artifacts]
#
#  BatchNorm2d normalises the 2D feature maps (not just 1D vectors like BN1d)

class CNN_AE_2(nn.Module):
    """Deep CNN Autoencoder | bottleneck=64 | BatchNorm2d throughout | Best model"""

    def __init__(self, bottleneck=64, drop=0.3):
        # drop=0.3: zero out 30% of neurons during training (higher regularisation)
        super().__init__()
        self.bottleneck = bottleneck

        self.encoder_conv = nn.Sequential(
            # First conv: stride=1 so spatial size stays 28×28, but channels increase 1→16
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),    # BN2d normalises the 16 channel feature maps
            nn.LeakyReLU(0.1),

            # Second conv: stride=2 halves spatial size 28→14
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            # Third conv: stride=2 halves spatial size 14→7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            # Fourth conv: stride=2, 7→4 (with padding)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),   # output: (B, 128, 4, 4)
        )

        # After conv: (B, 128, 4, 4) → flatten → (B, 2048) → linear → (B, 64)
        self.enc_fc = nn.Sequential(
            nn.Dropout(drop),                    # drop 30% of 2048 values
            nn.Linear(128 * 4 * 4, bottleneck),  # 2048 → 64
        )

        # Decoder FC: 64 → 2048
        self.dec_fc = nn.Sequential(
            nn.Linear(bottleneck, 128 * 4 * 4),  # 64 → 2048
            nn.LeakyReLU(0.1),
            nn.Dropout(drop),
        )

        self.decoder_conv = nn.Sequential(
            # ConvTranspose2d upsamples: (B,128,4,4) → (B,64,7,7)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            # (B,64,7,7) → (B,32,14,14)
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            # (B,32,14,14) → (B,16,28,28)
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            # Final regular conv (not transposed) to go from 16 channels to 1 cleanly
            nn.Conv2d(16, 1, kernel_size=3, padding=1),   # (B,16,28,28) → (B,1,28,28)
            nn.Sigmoid(),   # pixel values in [0,1]
        )

    def forward(self, x):
        h  = self.encoder_conv(x)                        # conv layers: (B,1,28,28)→(B,128,4,4)
        z  = self.enc_fc(h.view(x.size(0), -1))          # flatten+linear: 2048→64
        h2 = self.dec_fc(z).view(-1, 128, 4, 4)          # expand: 64→2048→reshape(128,4,4)
        return self.decoder_conv(h2), z                   # deconv to (B,1,28,28)


# ── SECTION 6: UTILITY FUNCTIONS ────────────────────────────────────────────────

def count_params(model):
    """Count the total number of trainable parameters in a model"""
    return sum(
        p.numel()                        # p.numel() = total elements in this parameter tensor
        for p in model.parameters()     # iterate over all parameter tensors in the model
        if p.requires_grad               # only count parameters that will be updated during training
    )


# Instantiate all 4 models (create actual objects from the class definitions above)
models_dict = {
    "ANN-1 (FC, BN=32)":   ANN_AE_1(bottleneck=32),  # simple shallow ANN
    "ANN-2 (FC, BN=64)":   ANN_AE_2(bottleneck=64),  # deep ANN with regularisation
    "CNN-1 (Conv, BN=32)": CNN_AE_1(bottleneck=32),  # lightweight CNN
    "CNN-2 (Conv, BN=64)": CNN_AE_2(bottleneck=64),  # deep CNN — our best model
}

# Print architecture summary
print(f"\n{'Model':<25} {'Type':<6} {'Bottleneck':>10} {'Parameters':>14}  Compression")
print("─" * 70)
for name, model in models_dict.items():
    t   = "ANN" if "ANN" in name else "CNN"    # model type
    bn  = model.bottleneck                      # bottleneck size
    p   = count_params(model)                   # total parameters
    cx  = round(784 / bn)                       # compression ratio (784 / bottleneck)
    print(f"{name:<25} {t:<6} {bn:>10} {p:>14,}  {cx}×")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
#
#  TRAINING LOOP EXPLAINED:
#  For each epoch (full pass through dataset):
#    For each batch of images:
#      1. Forward pass  : feed images through model → get reconstruction
#      2. Compute loss  : MSE(reconstruction, original) — how different are they?
#      3. Backward pass : compute gradients (how to adjust each weight to reduce loss)
#      4. Optimiser step: update weights using the gradients
#  Repeat for val set (no gradient updates, just measure loss)
#
#  LOSS FUNCTION: MSE (Mean Squared Error)
#  MSE = mean((original_pixel - reconstructed_pixel)²) for all pixels
#  It's 0 when reconstruction = original, and increases as they differ
#
#  OPTIMISER: Adam
#  Adam is an adaptive gradient descent optimiser — it adjusts learning rates per-parameter
#  Better than plain SGD for most deep learning tasks
#
#  LEARNING RATE SCHEDULER: ReduceLROnPlateau
#  If validation loss stops improving for `patience` epochs, halve the learning rate
#  This helps squeeze out extra performance near convergence
#
#  EARLY STOPPING:
#  If val loss doesn't improve for `patience` epochs, stop training early
#  Prevents wasting time on a model that's no longer improving


def train_one_epoch(model, loader, optimizer, criterion):
    """Run one full pass through the training set, updating weights"""
    model.train()   # set model to TRAINING mode (enables Dropout, BatchNorm behaves differently)
    total_loss = 0.0

    for imgs, _ in loader:
        # imgs: (batch_size, 1, 28, 28) — batch of images
        # _   : labels (ignored — autoencoders don't use labels during training!)
        imgs = imgs.to(DEVICE)   # move images to GPU (or keep on CPU if no GPU)

        optimizer.zero_grad()   # IMPORTANT: clear gradients from previous batch
                                 # (PyTorch accumulates gradients by default)

        recon, _ = model(imgs)   # FORWARD PASS: feed images through model
                                  # recon: reconstructed images, _: latent codes (not needed here)

        loss = criterion(recon, imgs)   # compute MSE loss between reconstruction and original
                                         # criterion is nn.MSELoss() defined in train_model()

        loss.backward()    # BACKWARD PASS: compute gradients via chain rule (backpropagation)
                           # Each weight gets a gradient: how much does loss change if we tweak this weight?

        optimizer.step()   # UPDATE WEIGHTS: move each weight slightly in the direction that reduces loss
                           # Adam uses the gradients + momentum + adaptive rates to do this smartly

        total_loss += loss.item() * imgs.size(0)
        # loss.item() converts tensor loss to plain Python float
        # multiply by batch size because we want total loss (not average), for correct epoch-level average

    return total_loss / len(loader.dataset)   # return average loss per image for this epoch


@torch.no_grad()   # decorator: tells PyTorch not to track gradients in this function
                   # saves memory and speeds up inference (we don't need gradients during evaluation)
def evaluate(model, loader, criterion):
    """Evaluate model on a dataset without updating weights"""
    model.eval()   # set model to EVALUATION mode (disables Dropout, BatchNorm uses running stats)
    total_loss = 0.0

    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        recon, _ = model(imgs)          # forward pass only (no backward needed)
        loss = criterion(recon, imgs)   # compute loss
        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def train_model(model, name, epochs=EPOCHS, lr=LR, patience=7):
    """Full training loop with early stopping and LR scheduling"""
    model.to(DEVICE)   # move all model parameters to GPU (or keep on CPU)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Adam: the most popular DL optimiser
    # model.parameters(): all the weights/biases that Adam will update
    # lr=0.001: initial learning rate

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,    # the optimiser whose LR we want to reduce
        mode='min',   # 'min' means "reduce LR when the monitored metric stops decreasing"
        patience=3,   # wait 3 epochs of no improvement before reducing
        factor=0.5,   # multiply LR by 0.5 (halve it) when triggered
        verbose=False # don't print every LR change (we print manually below)
    )

    criterion = nn.MSELoss()   # MSE loss function: mean((x - x_hat)^2)
                                # built into PyTorch, works on any shape tensors

    train_losses, val_losses = [], []   # lists to store loss history for plotting
    best_val    = float("inf")          # best validation loss seen so far (start at infinity)
    pat_cnt     = 0                     # counter for early stopping patience
    best_state  = None                  # will store the best model weights

    print(f"\n{'═'*60}")
    print(f"  [{name}]   Params: {count_params(model):,}")
    print(f"{'═'*60}")
    t0 = time.time()   # record start time

    for epoch in range(1, epochs + 1):   # loop from epoch 1 to epochs (inclusive)

        trn = train_one_epoch(model, train_loader, optimizer, criterion)  # train
        val = evaluate(model, val_loader, criterion)                       # validate
        scheduler.step(val)   # tell scheduler the new val loss — may trigger LR reduction

        train_losses.append(trn)   # save training loss for this epoch
        val_losses.append(val)     # save validation loss for this epoch

        if val < best_val:
            # new best model found!
            best_val   = val                                                          # update best
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} # save weights
            # state_dict() = dictionary of all parameter names → tensors
            # .cpu().clone() = copy to CPU (saves GPU memory), clone() = deep copy not reference
            pat_cnt = 0   # reset patience counter
        else:
            pat_cnt += 1  # one more epoch without improvement

        # print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']   # current learning rate
            print(f"  Ep {epoch:3d}/{epochs}  "
                  f"train={trn:.5f}  val={val:.5f}  lr={lr_now:.1e}")

        if pat_cnt >= patience:
            # early stopping triggered: no improvement for `patience` epochs
            print(f"  ⏹ Early stop @ epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"  ✓ Best val MSE: {best_val:.5f}  |  Time: {elapsed:.1f}s")
    model.load_state_dict(best_state)   # restore model to its best checkpoint
    return train_losses, val_losses, best_val


# ── RUN TRAINING ────────────────────────────────────────────────────────────────
print("\n\n🚀 Starting Training for all 4 Models...")
history_dict = {}   # stores (train_losses, val_losses, best_val) for each model

for name, model in models_dict.items():
    trn, val, best = train_model(model, name)   # train this model
    history_dict[name] = (trn, val, best)       # save its history

print("\n✅ All models trained!")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()   # no gradients needed for inference/plotting
def plot_reconstructions(model, name, loader, n=10):
    """Show input images (top row) vs reconstructed images (bottom row)"""
    model.eval()
    imgs, _ = next(iter(loader))    # get one batch from the loader
    imgs = imgs[:n].to(DEVICE)      # take first n images, move to device
    recon, _ = model(imgs)          # run through model
    imgs, recon = imgs.cpu(), recon.cpu()   # move back to CPU for matplotlib

    fig, axes = plt.subplots(2, n, figsize=(n * 1.7, 3.6))
    # 2 rows (input + reconstruction), n columns (one per image)
    fig.suptitle(f"[{name}]  Top: Input  |  Bottom: Reconstruction",
                 fontsize=11, fontweight='bold', y=1.02)

    for i in range(n):
        axes[0, i].imshow(imgs[i].squeeze(),  cmap='gray', vmin=0, vmax=1)  # original
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray', vmin=0, vmax=1)  # reconstruction
        axes[0, i].axis('off')   # hide axis ticks/labels
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Input\n",  rotation=0, labelpad=48, fontsize=9, va='center')
    axes[1, 0].set_ylabel("Recon.\n", rotation=0, labelpad=48, fontsize=9, va='center')

    plt.tight_layout()
    path = f"{OUT_DIR}/recon_{name.replace(' ','_').replace('/','-')}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved → {path}")


# Generate reconstruction plots for all models
for name, model in models_dict.items():
    plot_reconstructions(model, name, test_loader)


@torch.no_grad()
def plot_bottleneck_comparison(models_dict, loader, n=6):
    """Show same n images reconstructed by ALL 4 models side by side"""
    imgs, _ = next(iter(loader))
    sample = imgs[:n].to(DEVICE)   # same 6 images for all models

    recons = {}
    for name, model in models_dict.items():
        model.eval()
        r, _ = model(sample)        # reconstruct with this model
        recons[name] = r.cpu()      # store on CPU

    rows = ["Original Input"] + list(models_dict.keys())  # row labels
    data = [sample.cpu()] + [recons[k] for k in models_dict]  # row data

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.8, len(rows) * 2.0))
    fig.suptitle("Bottleneck Effect — Same Images Through All Models",
                 fontweight='bold', fontsize=11)

    for r, (label, row_data) in enumerate(zip(rows, data)):
        for c in range(n):
            axes[r, c].imshow(row_data[c].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(label, rotation=0, labelpad=90, fontsize=8, va='center')

    plt.tight_layout()
    path = f"{OUT_DIR}/bottleneck_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {path}")


plot_bottleneck_comparison(models_dict, test_loader)


def plot_loss_curves(history_dict):
    """Plot training AND validation loss curves for all models on one figure"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))   # 1 row, 2 columns
    colors = plt.cm.tab10.colors    # 10 distinct colours from matplotlib's tab10 palette
    styles = ['-', '--', '-.', ':'] # different line styles to distinguish models

    for i, (name, (trn, val, _)) in enumerate(history_dict.items()):
        c  = colors[i]                        # colour for this model
        ep = range(1, len(trn) + 1)           # epoch numbers 1,2,...,N
        axes[0].plot(ep, trn, color=c, ls=styles[i], lw=2, label=name)  # training loss
        axes[1].plot(ep, val, color=c, ls=styles[i], lw=2, label=name)  # validation loss

    for ax, title in zip(axes, ["Training Loss (MSE)", "Validation Loss (MSE)"]):
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE", fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)   # show legend box
        ax.grid(alpha=0.3)      # light grid lines

    plt.suptitle("Loss Curve Comparison — All Autoencoder Models",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT_DIR}/loss_curves_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {path}")


plot_loss_curves(history_dict)


@torch.no_grad()
def collect_latents(model, loader, n_batches=25):
    """Collect latent vectors z and their class labels from the dataset"""
    model.eval()
    zs, lbls = [], []

    for i, (imgs, labels) in enumerate(loader):
        if i >= n_batches:
            break   # stop after n_batches to avoid processing the whole test set
        _, z = model(imgs.to(DEVICE))   # run forward pass, get latent code z
        zs.append(z.cpu().numpy())      # convert tensor to numpy array, collect
        lbls.append(labels.numpy())     # collect labels too

    return np.concatenate(zs), np.concatenate(lbls) - 1
    # np.concatenate: stack all collected arrays into one large array
    # - 1 converts labels from 1-26 to 0-25 (0-indexed)


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")   # for colourbqr tick labels


def plot_latent(model, name, loader, method='pca', n_batches=25):
    """2D scatter plot of the latent space using PCA or t-SNE"""
    zs, labels = collect_latents(model, loader, n_batches)
    # zs: (N, bottleneck) — e.g. (6400, 64) for 25 batches × 256 images × 64 dims

    if method == 'pca':
        # PCA: Principal Component Analysis
        # Finds the directions of maximum variance in the data
        # Projects high-dimensional data to 2D by keeping the 2 most important directions
        # Fast but may miss non-linear structure
        emb    = PCA(n_components=2, random_state=SEED).fit_transform(zs)
        suffix = "PCA"
    else:
        # t-SNE: t-distributed Stochastic Neighbour Embedding
        # Non-linear method: preserves local neighbourhood structure
        # Shows clusters much more clearly than PCA, but is slower
        idx = np.random.choice(len(zs), min(3000, len(zs)), replace=False)
        # subsample to 3000 points max (t-SNE is O(n²) so full set would be slow)
        zs, labels = zs[idx], labels[idx]
        emb = TSNE(
            n_components=2,      # project to 2D
            random_state=SEED,
            perplexity=30,       # roughly = expected cluster size
            n_iter=1000,         # number of optimisation iterations
            learning_rate='auto',
            init='pca'           # initialise with PCA (more stable than random init)
        ).fit_transform(zs)
        suffix = "t-SNE"

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],   # x and y coordinates
        c=labels,                # colour each point by its class
        cmap='tab20',            # 20-colour map (enough for 26 letters)
        s=5,                     # dot size
        alpha=0.65,              # transparency (so overlapping points visible)
        linewidths=0             # no dot outlines
    )

    cbar = plt.colorbar(sc, ax=ax, ticks=range(26))  # colour legend on right
    cbar.set_ticklabels(LETTERS)   # label ticks A, B, C, ... Z
    cbar.set_label("Letter Class", fontsize=10)

    ax.set_title(f"[{name}] Latent Space — {suffix}", fontweight='bold', fontsize=12)
    ax.set_xlabel(f"{suffix} dim 1")
    ax.set_ylabel(f"{suffix} dim 2")
    plt.tight_layout()

    path = f"{OUT_DIR}/latent_{name.replace(' ','_')}_{method}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {path}")


# PCA for all 4 models
for name, model in models_dict.items():
    plot_latent(model, name, test_loader, method='pca')

# t-SNE for the two best-performing models
for name in ["CNN-2 (Conv, BN=64)", "ANN-2 (FC, BN=64)"]:
    print(f"Running t-SNE for {name}...")
    plot_latent(models_dict[name], name, test_loader, method='tsne')


def plot_summary(history_dict, models_dict):
    """Side-by-side bar charts comparing val loss and parameter count"""
    names  = list(history_dict.keys())
    vals   = [v for _, _, v in history_dict.values()]   # extract best val loss for each model
    params = [count_params(models_dict[n]) / 1e3 for n in names]  # params in thousands
    x      = np.arange(len(names))   # array [0, 1, 2, 3] for positioning bars

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    colors4 = plt.cm.tab10.colors[:4]  # first 4 colours

    # LEFT: validation loss (lower = better reconstruction)
    bars1 = ax1.barh(x, vals, color=colors4)   # horizontal bar chart
    ax1.set_yticks(x); ax1.set_yticklabels(names)
    ax1.set_xlabel("Best Validation MSE Loss")
    ax1.set_title("Reconstruction Quality (lower = better)", fontweight='bold')
    for bar, v in zip(bars1, vals):
        ax1.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height()/2,
                 f"{v:.5f}", va='center', fontsize=9)   # label each bar with its value
    ax1.grid(axis='x', alpha=0.3)

    # RIGHT: parameter count (smaller = more efficient)
    bars2 = ax2.barh(x, params, color=colors4)
    ax2.set_yticks(x); ax2.set_yticklabels(names)
    ax2.set_xlabel("Trainable Parameters (thousands)")
    ax2.set_title("Model Size", fontweight='bold')
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height()/2,
                 f"{p:.0f}K", va='center', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle("All Models — Comparison Summary", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f"{OUT_DIR}/summary_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()

    # print text summary table
    best_mse = min(vals)
    print(f"\n{'Model':<25} {'Params':>8} {'Best Val MSE':>14} {'Compression':>12}  Rank")
    print("─" * 72)
    ranked = sorted(history_dict.items(), key=lambda x: x[1][2])  # sort by val loss
    for rank, (name, (_, _, v)) in enumerate(ranked, 1):
        p   = count_params(models_dict[name])
        bn  = models_dict[name].bottleneck
        cx  = round(784 / bn)
        rel = "★" if rank == 1 else f"+{(v - best_mse)*100:.2f}%"
        print(f"{name:<25} {p//1000:>7}K {v:>14.5f} {cx:>11}×   #{rank} {rel}")


plot_summary(history_dict, models_dict)


# ── SECTION 9: SAVE MODEL WEIGHTS ───────────────────────────────────────────────
# Save trained model weights so you can reload them later without retraining

for name, model in models_dict.items():
    fname = name.replace(' ', '_').replace('/', '-').replace('=', '') + ".pt"
    # .pt is the standard extension for PyTorch saved files
    path  = f"{OUT_DIR}/{fname}"

    torch.save({
        'model_state_dict': model.state_dict(),  # the actual weights (a dict of tensors)
        'bottleneck': model.bottleneck,           # save architecture info alongside weights
        'val_loss': history_dict[name][2],        # best validation loss achieved
        'architecture': name,                     # human-readable name
    }, path)
    print(f"Saved: {path}")

print(f"\n✅ All done! Outputs saved to: {OUT_DIR}/")


# ── SECTION 10: RESULTS SUMMARY ─────────────────────────────────────────────────
#
#  WHAT THE RESULTS TELL US:
#
#  1. CNN-2 is the best model (val MSE = 0.00192)
#     → Deep CNNs with BatchNorm learn the best compressed representations
#
#  2. CNN-1 outperforms ANN-2 despite having 25× fewer parameters
#     → Convolutions are extremely efficient for image data because:
#       a) Spatial locality: a 3×3 filter sees pixels that are actually neighbours
#       b) Weight sharing: the same filter detects edges/curves across the whole image
#       c) ANNs treat pixel (0,0) and pixel (27,27) as equally related — which is wrong
#
#  3. Bottleneck 64 always outperforms bottleneck 32
#     → More dimensions = more information retained = better reconstruction
#     → But it's a trade-off: 64-dim = 12× compression vs 32-dim = 24× compression
#
#  4. BatchNorm + Dropout improve stability and generalisation
#     → The loss curves for ANN-2 and CNN-2 are smoother (less noisy)
#     → Smaller gap between train and val loss = less overfitting
#
#  5. Latent space structure (t-SNE plots):
#     → Even though the model never saw labels during training,
#       similar letters cluster together in the latent space!
#     → CNN-2 shows the cleanest separation because its representations
#       are more structured — convolutions enforce geometric reasoning
