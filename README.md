# AttnGAN-Inspired Fine-Grained Bird Image Synthesis

An AttnGAN-inspired text-to-image synthesis project for generating fine-grained bird species images using Generative Adversarial Networks (GANs), Bidirectional LSTM text encoding, and word-level attention mechanisms.

This project was developed using the **CUB-200-2011 dataset** and compares a baseline GAN architecture against an improved attention-based model inspired by the original AttnGAN paper.

---

## Overview

Text-to-image synthesis is a challenging task in deep learning where a model generates realistic images from natural language descriptions.

This project implements:

* A **baseline text-conditioned GAN**
* An **improved AttnGAN-style architecture**
* Automatic caption generation for the CUB dataset
* Word-level cross-attention
* Checkerboard-free upsampling generator

The goal is to evaluate how attention mechanisms improve image quality and semantic consistency under constrained training conditions.

---

## Features

* Text-to-image bird generation
* BiLSTM text encoder
* Word-level cross-attention
* Upsample + Conv generator (checkerboard-free)
* Automatic caption generation pipeline
* PyTorch implementation
* CUB-200-2011 dataset support
* Baseline vs Attention model comparison

---

## Architecture

### Baseline Model

* Mean-pooled word embeddings
* ConvTranspose2d generator
* Sentence-level conditioning only

### Improved Model

* Bidirectional LSTM text encoder
* Word-level attention mechanism
* Upsample + Conv architecture
* Improved image sharpness and diversity

---

## Dataset

This project uses the:

### CUB-200-2011 Dataset

* 11,788 bird images
* 200 bird species
* Fine-grained classification dataset

Subset used in experiments:

* 200 images
* 10 bird classes
* 64×64 resolution

---

## Experimental Results

| Model            | Pixel Diversity ↑ | Sharpness ↑ | Checkerboard |
| ---------------- | ----------------- | ----------- | ------------ |
| Baseline         | 0.1823            | 0.0412      | Yes          |
| Improved AttnGAN | 0.2044            | 0.0509      | No           |

### Improvements

* +12.1% Pixel Diversity
* +23.5% Sharpness
* Checkerboard artifacts eliminated

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Google Colab

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Rewanooo22/AttnGAN.git
cd AttnGAN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run training:

```bash
python train.py
```

---

## Project Structure

```text
AttnGAN/
│
├── data/
├── models/
├── captions/
├── outputs/
├── train.py
├── generator.py
├── discriminator.py
├── text_encoder.py
├── attention.py
├── requirements.txt
└── README.md
```

---

## Methodology

### Automatic Caption Generation

Since CUB-200-2011 lacks natural language captions, a template-based caption generation system was created using:

* Colors
* Patterns
* Sizes
* Background contexts
* Perch contexts

Example:

> "A small blue Laysan Albatross with spotted feathers perched on a branch in a forest."

---

## Attention Mechanism

The improved model uses word-level cross-attention:

[
\alpha_{i,j} = softmax(\gamma \cdot q_i^T w_j)
]

allowing image regions to attend selectively to relevant words in the caption.

---

## Limitations

* Small training subset
* 64×64 image resolution
* Template-generated captions
* Limited computational resources

---

## Future Work

* Train on full CUB dataset
* Higher image resolutions (128×128 / 256×256)
* Add DAMSM loss
* Use human-annotated captions
* Compute FID and Inception Score

---

## References

* Goodfellow et al. — GANs (2014)
* Xu et al. — AttnGAN (2018)
* Zhang et al. — StackGAN (2017)
* Odena et al. — Checkerboard Artifacts (2016)

---

## Author

**Rewan Hany**

---

## License

This project is for educational and research purposes.

