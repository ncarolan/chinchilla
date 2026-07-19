# chinchilla
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ncarolan/chinchilla/blob/main/chinchilla.ipynb)

An experiment reproducing Chinchilla-style ([Hoffmann et al 2022](https://arxiv.org/abs/2203.15556)) language model scaling laws in a "minimal" setup.

Everything is contained in a single Google Colab notebook and runs on a single (free-tier) NVIDIA T4 GPU. This is achieved by using small models (<10M parameters) and training on synthetic addition data. 

The project is based Vlad Feinberg's [blog post](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html):
>“Code a ~10M transformer using only jax, flax, optax in free colab using tpu. Hard code it to accept digits 0-9, space, +, =. Generate a dataset of simple up-to-3-digit numbers, have it learn addition. Should train quickly on T4 GPU (pad examples to fixed length). Derive Chinchilla laws for this; see how they differ for dense vs MoE architectures.”

The full experimental report, with plots and scaling exponents, is found at [nicholascarolan.com/chinchilla](https://www.nicholascarolan.com/chinchilla/).
