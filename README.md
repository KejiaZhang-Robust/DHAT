# Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment

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
