# Data-Argumentation-On-Low-Rate-DDoS

Low‑rate distributed denial‑of‑service (LR‑DDoS) attacks emit sparse, protocol‑compliant bursts that silently deplete edge resources while remaining below volumetric thresholds, leaving detection models hindered by severe class imbalance and scarce minority samples.
This study benchmarks five augmentation regimes: None, SMOTE, SMOTENC, conditional variational auto‑encoders (CVAE), and generative adversarial networks (GAN), across four mainstream classifiers (Logistic Regression, Multilayer Perceptron, Random Forest, XGBoost) on CICIoT2023 (binary) and CIC‑IDS2017 (three‑class) traffic. Early‑epoch checkpoints and ten random seeds are used to evaluate convergence speed and robustness.

SMOTENC delivers the most consistent gains, lifting minority‑class F1 from 0.55 to 0.94 in the 47:1 binary task and from 0.64 to 0.94 in the 400:1 multi‑class task, with negligible variance. CVAE attains F1 ≥ 0.99 by epoch 3 on the MLP but incurs an average 0.6 percentage point stability penalty. GAN augmentation remains statistically indistinguishable from the baseline, indicating insufficient synthetic diversity. Tree ensembles sustain F1 ≥ 0.995 regardless of augmentation, whereas linear models derive the greatest benefit from oversampling.

Contrary to the initial hypothesis that generative methods would dominate, classical mixed‑type oversampling (SMOTENC) proved the most reliable. CVAE offers a favourable speed–accuracy trade‑off for rapid retraining on resource‑constrained devices, while GANs require further refinement. Limitations include dependence on two public datasets and offline augmentation; future work will target online, energy‑aware generators and privacy‑preserving federated deployments.

Index Terms — IoT, edge computing, low‑rate DDoS, data augmentation, class imbalance.
