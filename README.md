# EMNIST Autoencoder - Image Compression & Reconstruction

> **Assignment:** Build autoencoder models from scratch in PyTorch on the EMNIST Letters dataset  
> **Dataset:** [EMNIST Letters](https://www.kaggle.com/datasets/crawford/emnist) - 26 handwritten letter classes, 28×28 greyscale  
> **Author:** Rishii Kumar Singh

---

## 📋 Task Overview

An **autoencoder** is a neural network trained to compress an image into a small vector (the *latent code*), and then reconstruct the original image from that code. The smaller the code, the more the model has "understood" the underlying structure of the data.

```
Input Image (784 pixels)
       │
   [Encoder]  ←── learns to compress
       │
  Latent z    ←── 32 or 64 numbers (the "code")
       │
   [Decoder]  ←── learns to reconstruct
       │
Reconstructed Image (784 pixels)

Training goal: minimise MSE(input, reconstruction)
```

---

## Models Built (All From Scratch)

| Model | Type | Bottleneck | Parameters | Val MSE |
|-------|------|-----------|-----------|---------|
| ANN-1 | Fully Connected | 32 | 477K | 0.01124 |
| ANN-2 | FC + BatchNorm + Dropout | 64 | 1,153K | 0.00715 |
| CNN-1 | Convolutional (Lightweight) | 32 | **45K** | 0.00584 |
| **CNN-2** | **Deep CNN + BN** | **64** | 459K | **0.00192 ★** |

**Key finding:** CNN-1 (45K params) beats ANN-2 (1,153K params) - **25× fewer parameters, better quality** - because convolutions exploit spatial structure.

---

## Repository Structure

```
EMNIST_Autoencoder/
│
├── autoencoder_emnist_commented.py   ← Full code with line-by-line beginner comments
├── autoencoder_emnist.ipynb          ← Jupyter notebook (with all outputs)
├── requirements.txt                  ← Python dependencies
│
├── outputs/                          ← All generated visualisations
│   ├── 0_dataset_samples.png         ← Sample images from EMNIST
│   ├── recon_ANN-1_*.png             ← Input vs reconstruction (ANN-1)
│   ├── recon_ANN-2_*.png             ← Input vs reconstruction (ANN-2)
│   ├── recon_CNN-1_*.png             ← Input vs reconstruction (CNN-1)
│   ├── recon_CNN-2_*.png             ← Input vs reconstruction (CNN-2)
│   ├── bottleneck_comparison.png     ← All 4 models on same images
│   ├── loss_curves_comparison.png    ← Train & val loss for all models
│   ├── latent_*_pca.png              ← 2D PCA of latent space
│   ├── latent_*_tsne.png             ← 2D t-SNE of latent space
│   └── summary_comparison.png        ← Final comparison bar charts
│
└── models/                           ← Saved model weights (.pt files)
    ├── ANN-1__FC__BN32_.pt
    ├── ANN-2__FC__BN64_.pt
    ├── CNN-1__Conv__BN32_.pt
    └── CNN-2__Conv__BN64_.pt
```

---

## 🚀 Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/EMNIST_Autoencoder.git
cd EMNIST_Autoencoder

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run as a script
python autoencoder_emnist_commented.py

# 3b. Or open the notebook
jupyter notebook autoencoder_emnist.ipynb
```

> **Note:** EMNIST dataset (~530 MB) downloads automatically on first run.  
> **GPU:** Works on any PyTorch-compatible GPU (sm_70+). Falls back to CPU automatically if not available.

---

## 📊 Results

### Reconstruction Quality
CNN-2 produces the sharpest reconstructions with clean stroke details even at 12× compression.

### Loss Curves
- CNN models converge much faster and to lower loss than ANN models
- BatchNorm (ANN-2, CNN-2) produces smoother curves with less noise
- All models trained for 40 epochs with Adam + ReduceLROnPlateau scheduler

### Latent Space (t-SNE)
Even though the model is **never given class labels during training**, the t-SNE plot shows clear letter clusters — the autoencoder has learned meaningful structure from pixel patterns alone. CNN-2 shows the cleanest separation.

---

## 🔧 Training Details

| Setting | Value |
|---------|-------|
| Loss | MSE (Mean Squared Error) |
| Optimiser | Adam (lr = 1e-3) |
| LR Scheduler | ReduceLROnPlateau (patience=3, ×0.5) |
| Early Stopping | Patience = 7 epochs |
| Epochs | 40 |
| Batch Size | 256 |
| Train/Val Split | 90% / 10% |

---

## 📚 Concepts Covered

- **Autoencoder architecture** (Encoder → Latent → Decoder)
- **Fully Connected (ANN) vs Convolutional (CNN)** autoencoders
- **Bottleneck / latent space** and compression ratio
- **BatchNorm, Dropout, LeakyReLU** regularisation
- **Reconstruction loss** (MSE)
- **Latent space visualisation** (PCA, t-SNE)
- **Early stopping** and **learning rate scheduling**
