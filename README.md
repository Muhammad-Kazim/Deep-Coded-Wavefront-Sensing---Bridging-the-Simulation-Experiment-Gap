# Deep Coded Wavefront Sensing (Deep CWFS): Bridging the Simulation–Experiment Gap

---

## 📌 Overview

This repository contains the official implementation of **Deep Coded Wavefront Sensing: Bridging the Simulation-Experiment Gap**, a learning-based framework for quantitative phase imaging using coded wavefront sensing (CWFS).

Deep CWFS leverages **wave-optical simulation** to train neural networks entirely on synthetic data of microspheres, enabling accurate phase reconstruction on real-world microscopic specimens—**without requiring experimental training data**.

---

## 🚀 Key Contributions

- 🔬 **Wave-optical forward model** for CWFS  
- 📦 **SynthBeads dataset**: high-fidelity synthetic CWFS data  
- 🧠 **Deep CWFS**: neural network–based phase retrieval  
- 🌉 **Simulation-to-experiment generalization** without domain gap  
- 🧫 Successful application to:
  - Synthetic cell data (SynthCell)
  - Experimental microbeads
  - Complex biological HEK Cells

---

## 📄 Paper

**Deep Coded Wavefront Sensing: Bridging the Simulation–Experiment Gap**  
S. M. Kazim, P. Müller, and I. Ihrke  
NeurIPS 2025 Workshop: *Learning to Sense*

📎 [Read the paper](https://openreview.net/pdf?id=4dQlpj6LHc)

---

## 🧠 Abstract

Coded wavefront sensing (CWFS) is a recent computational quantitative phase imaging technique that enables one-shot phase retrieval of biological and other phase specimens. CWFS is readily integrable with standard laboratory microscopes and does not require specialized labor for its usage. The CWFS phase retrieval method is inspired by optical flow, but uses conventional optimization techniques. A main reason for this is the lack of publicly available datasets for CWFS, which prevents researchers from using deep neural networks in CWFS. In this paper, we present a forward model that utilizes wave optics to generate SynthBeads: a CWFS dataset obtained by modeling the complete experimental setup, including wave propagation through refractive index (RI) volumes of spherical microbeads, a standard microscope, and the phase mask, which is a key component of CWFS, with high fidelity. We show that our forward model enables deep CWFS, where pre-trained optical flow networks finetuned on SynthBeads successfully generalize to our SynthCell dataset, experimental microbead measurements, and, remarkably, complex biological specimens, providing quantitative phase estimates and thereby bridging the simulation-experiment gap.

---

## 📦 Datasets

**SynthBeads (Synthetic Microbeads)**
- Generated using a wave-optical CWFS simulator
- Includes:
  - Reference speckle images
  - Distorted measurements
  - Ground-truth optical flow / phase

**SynthCell (Synthetic Cells)**
- Used to evaluate generalization beyond training distribution

---

## ⚙️ **Installation**
```bash
git clone https://github.com/Muhammad-Kazim/Deep-Coded-Wavefront-Sensing---Bridging-the-Simulation-Experiment-Gap.git
cd Deep-Coded-Wavefront-Sensing---Bridging-the-Simulation-Experiment-Gap
conda env create -n custom_env_name -f environment.yml
conda activate custom_env_name
pip install -e .
```

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{kazim2025deepcwfs,
  title={Deep Coded Wavefront Sensing: Bridging the Simulation-Experiment Gap},
  author={Kazim, Syed Muhammad and Müller, Patrick, and Ihrke, Ivo},
  booktitle={NeurIPS Workshop on Learning to Sense},
  year={2025},
  url={https://openreview.net/pdf?id=4dQlpj6LHc}
}
```
