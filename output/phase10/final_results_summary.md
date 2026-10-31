# Final Results Summary: ICAIN 2026 Paper ID 1567

### 1. Main Corel-1K Benchmark Performance (899 Gallery, 100 Queries)
- **M1 (Colour Only):** P@10 = 0.6190, R@10 = 0.0689, F1@10 = 0.1240, mAP = 0.4045
- **M2 (Texture Only):** P@10 = 0.4680, R@10 = 0.0521, F1@10 = 0.0937, mAP = 0.3508
- **M3 (Equal Fusion 0.5/0.5):** P@10 = 0.6990, R@10 = 0.0778, F1@10 = 0.1400, mAP = 0.4650
- **M4 (Heuristic Fixed 0.7/0.3):** P@10 = 0.6730, R@10 = 0.0749, F1@10 = 0.1348, mAP = 0.4337
- **M5 (Proposed Adaptive MLP):** P@10 = **0.7100**, R@10 = **0.0790**, F1@10 = **0.1422**, mAP = **0.4798**
- **M6 (Mean Weight Control):** P@10 = 0.6960, R@10 = 0.0774, F1@10 = 0.1394, mAP = 0.4605

### 2. Inferential Statistical Comparisons Against Proposed M5
- **M5 vs D1 (LBP):** $\Delta = +0.1064$, $p = 6.61 \times 10^{-9}$ (Holm-adj $p < 0.0001$, **Statistically Significant**, 81 wins / 19 losses)
- **M5 vs M4 (Fixed 0.7/0.3):** $\Delta = +0.0461$, $p = 5.35 \times 10^{-8}$ (Holm-adj $p = 0.0001$, **Statistically Significant**, 76 wins / 24 losses)
- **M5 vs M3 (Equal 0.5/0.5):** $\Delta = +0.0148$, $p = 0.3259$ (Holm-adj $p = 0.6519$, **Non-Significant**, 51 wins / 43 losses)
- **M5 vs M6 (Mean Control):** $\Delta = +0.0193$, $p = 0.5543$ (Holm-adj $p = 0.6519$, **Non-Significant**, 53 wins / 47 losses)
- **M5 vs Deep Models (ResNet/EffNet/CLIP):** Deep models achieve $0.88 - 0.95$ mAP and are statistically superior ($p < 10^{-15}$).

### 3. Resource and Edge Scalability Profile
- **Feature Storage:** 52 bytes/image (13 float32). Corel gallery = 45.65 KB (cache-resident).
- **Extraction Latency:** 0.45 ms/image (~2,222 images/sec on CPU).
- **MLP Inference:** 0.0827 ms/query (450 parameters, 1.76 KB).
- **Large Gallery Scaling:** $\mathcal{O}(N)$ strictly linear; $N=7,000$ ranking takes 0.477 ms on CPU.

### 4. Zero-Shot Oxford-IIIT Pet Transfer (37 Breeds)
- **M5 (Proposed):** mAP = 0.033332, P@10 = 0.0546
- **M6 (Mean Control):** mAP = 0.033155, P@10 = 0.0584
- **M3 (Equal Fusion):** mAP = 0.033275, P@10 = 0.0568
- Transparently reconciles earlier manuscript draft numbers ($0.033832$ vs $0.033715$).
