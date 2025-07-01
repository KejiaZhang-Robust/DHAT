<div align="center">
  <h2 style="font-size: 36px; font-weight: bold; color: #333;">Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment</h2>
</div>

<div align="center" style="margin-top: 20px;">
  <!-- arXiv Badge -->
  <a href="https://arxiv.org/abs/2408.06079">
    <img src="https://img.shields.io/badge/arXiv-2408.06079-b31b1b?style=flat-square" alt="arXiv" style="margin: 0 0px;" />
  </a>
  <!-- License Badge -->
  <img alt="GitHub License" src="https://img.shields.io/github/license/KejiaZhang-Robust/2408.06079?style=flat-square" style="margin: 0 0px;">
  <!-- Language Badge -->
  <img alt="Language" src="https://img.shields.io/github/languages/top/KejiaZhang-Robust/2408.06079?style=flat-square&color=9acd32" style="margin: 0 5px;">
</div>

📌 **Accepted at ICCV 2025**

This repository contains the official PyTorch implementation of our ICCV 2025 paper:  
**"Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment"**.

---

## 🏋️‍♂️ Training

1. Modify the training configuration:

```bash
configs_train.yml
```

2. Start training:

```bash
python train.py
```

---

## 🧪 Robustness Evaluation

1. Edit the testing configuration:

```bash
configs_test.yml
```

2. Launch evaluation:

```bash
python test_robust.py
```

---

## 📄 Citation

```bibtex
@inproceedings{zhang2025dhat,
  title     = {Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment},
  author={Kejia Zhang and Juanjuan Weng and Shaozi Li and Zhiming Luo},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---
