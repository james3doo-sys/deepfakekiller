### OOD 일반화 (`ComplexDataLab/OpenFake` core/test, 학습에 안 씀)
| 조합 | AUC | EER | n |
|---|---|---|---|
| CNN only | 0.6314 | 0.4150 | 1000 |
| ViT only | 0.5947 | 0.4390 | 1000 |
| CNN + ViT (0.45 / 0.45) | 0.6337 | 0.4081 | 1000 |
| CNN + ViT + ELA (deepfakekiller.py 기본값) | 0.6274 | 0.4120 | 1000 |
