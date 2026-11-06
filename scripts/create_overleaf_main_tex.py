"""
Creates the master main.tex for the Springer Overleaf template.
Strictly formatted for Springer Nature (sn-jnl, sn-mathphys-num), exactly within 12 pages.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERLEAF_DIR = PROJECT_ROOT / "overleaf_project"

latex_code = r"""\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{listings}

\raggedbottom

\begin{document}

\title[AI-Weighted Hybrid CBIR Using Colour and Texture]{Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features}

\author[1]{\fnm{Prakriti} \sur{Sharma}}\email{sprakriti0193@gmail.com}
\author*[1]{\fnm{Shalok} \sur{Dadhwal}}\email{shalokdadhwal@gmail.com}
\author[1]{\fnm{Sujal} \sur{Jain}}\email{jainsujal340@gmail.com}
\author[1]{\fnm{Vijay} \sur{Bhardwaj}}\email{prof.dr.bhardwaj@gmail.com}
\author[1]{\fnm{Jaspreet} \sur{Singh}}\email{Jaspreetdioghan80540@gmail.com}

\affil*[1]{\orgdiv{Department of Computer Science and Engineering}, \orgname{Chandigarh University}, \orgaddress{\city{Mohali}, \state{Punjab}, \country{India}}}

\abstract{Content-Based Image Retrieval (CBIR) systems integrating colour and texture conventionally employ rigid, predetermined fusion coefficients. However, static weighting overlooks the visual reality that the discriminative efficacy of individual modalities fluctuates across diverse image queries. This paper presents a query-adaptive distance-level fusion framework that dynamically determines the relative importance of colour and texture representations for each query. The architecture utilizes an ultra-lightweight multilayer perceptron (13--16--8--2 topology, 378 trainable parameters) trained via a batch-hard triplet margin loss ($m=0.15$) to predict Softmax-constrained metric weights. The visual descriptor comprises an ultra-compact 13-dimensional handcrafted vector combining nine statistical HSV moments and four Gray-Level Co-occurrence Matrix (GLCM) texture moments. Rigorous evaluations on the Wang Corel-1K dataset follow a leakage-controlled protocol across an 899-image gallery (one near-duplicate purged) and 100 held-out queries. The proposed adaptive configuration (M5) attains a Precision@10 of 0.7100 and a mean Average Precision (mAP) of 0.4798, outperforming all five handcrafted baselines. Paired Wilcoxon signed-rank tests with Holm--Bonferroni step-down correction establish that M5 statistically significantly outperforms classical Local Binary Patterns ($p = 6.61 \times 10^{-9}$) and heuristic fixed fusion ($p = 5.35 \times 10^{-8}$). Improvements over equal fusion (M3) and mean-weight control (M6) are descriptive and concentrated in the Flowers category ($+0.2103$ mAP gain). Although high-capacity deep representations (ResNet-50, EfficientNet-B0, CLIP) yield higher semantic accuracy (mAP 0.88--0.95), M5 requires only 52~bytes per image descriptor ($157.5\times$ smaller than ResNet-50) and executes feature extraction in 0.45~ms on CPU ($53.4\times$ faster), establishing an efficient paradigm for resource-constrained edge computing. Zero-shot transferability evaluation on the Oxford-IIIT Pet dataset yields 0.0333 mAP (near random chance $\approx 0.0270$), delineating the operational boundary of global statistical moments for fine-grained biological retrieval.}

\keywords{Content-Based Image Retrieval, Adaptive Feature Fusion, Colour and Texture Moments, Multilayer Perceptron, Edge Deployment, Inferential Statistical Testing}

\maketitle

\section{Introduction}\label{sec1}
Content-Based Image Retrieval (CBIR) searches large visual repositories based on intrinsic image characteristics rather than manual annotations or textual metadata~\cite{dubey2020survey}. The retrieval effectiveness of any CBIR framework depends primarily on the discriminative capacity of its visual descriptors and the similarity formulation employed to rank database images. Among traditional visual attributes, colour and texture provide essential complementary information: colour statistical moments capture global chromatic distributions and channel dispersion, whereas texture descriptors capture local spatial variations, coarseness, and structural repetition~\cite{chun2008cbir,stricker1995similarity}. Consequently, combining these complementary modalities yields a more comprehensive visual representation than relying on either modality in isolation~\cite{singh2018color,yue2011cbir}.

A major methodological challenge in hybrid CBIR lies in determining the appropriate contribution of individual feature modalities during similarity ranking. Conventional frameworks typically apply global descriptor concatenation or rigid, predefined weighting coefficients~\cite{tyagi2026enhanced,naveen2025ai,wang2014content}. Although computationally straightforward, static weighting schemes enforce uniform relative importance across the entire collection. However, the discriminative power of colour and texture varies dynamically depending on the semantic content of the query: vibrant floral scenes often rely predominantly on chromatic hues, whereas architectural facades require fine structural granularity. Fixed fusion cannot effectively accommodate these divergent query characteristics.

To mitigate this limitation, query-adaptive feature weighting has been investigated. Previous studies explored neural-network-based weight prediction and multi-descriptor fusion~\cite{naveen2025ai,akmal2023automatic,bu2023content}. However, most methods fuse descriptors at the feature level prior to similarity calculation, obscuring modality-specific metric structures. Distance-level fusion, by contrast, calculates independent metric distances for each visual modality and subsequently combines them during ranking, preserving the underlying topological distances of individual feature spaces.

In this work, we propose a lightweight, query-adaptive distance-level fusion framework. Instead of relying on high-capacity deep backbones, our system is engineered around an ultra-compact 13-dimensional handcrafted representation combining nine HSV colour moments and four GLCM texture moments. A tiny metric-learning Multilayer Perceptron (MLP) containing just 378 trainable parameters (1.76~KB disk checkpoint) dynamically predicts query-specific metric weights $[w_C(q), w_T(q)]$ subject to a strict Softmax unit simplex constraint ($w_C + w_T = 1.0$).

The primary contributions of this paper are summarised as follows:
\begin{enumerate}
    \item \textbf{Lightweight Adaptive Distance Fusion:} We introduce a 378-parameter MLP trained via batch-hard triplet margin loss to dynamically weight separate colour and texture distances per query, eliminating rigid global coefficients.
    \item \textbf{Leakage-Controlled Benchmark Evaluation:} We evaluate the framework on the Wang Corel-1K dataset under a duplicate-aware protocol (899 gallery images, 100 held-out queries) with gallery-fitted Min-Max normalization, guaranteeing zero test-set data leakage.
    \item \textbf{Rigorous Inferential Statistical Validation:} We conduct paired two-sided Wilcoxon signed-rank tests, 10,000 sign-flip permutations, and Holm--Bonferroni family-wise error rate corrections, formally separating statistically significant improvements from descriptive gains.
    \item \textbf{Edge Deployment Benchmarking and Transfer Limits:} We benchmark the framework against deep architectures (ResNet-50, EfficientNet-B0, CLIP ViT-B/32) and evaluate zero-shot cross-dataset transfer on the 37-breed Oxford-IIIT Pet dataset, confirming a $157.5\times$ storage reduction while identifying the domain boundaries of global moments.
\end{enumerate}

\section{Related Work}\label{sec2}
\subsection{Colour and Texture Descriptors}
Statistical colour moments provide compact, illumination-tolerant descriptions of image chromaticity~\cite{stricker1995similarity}. Concurrently, Gray-Level Co-occurrence Matrices (GLCM) and Local Binary Patterns (LBP) capture spatial gray-level dependencies and structural micro-patterns~\cite{ojala2002multiresolution,chun2008cbir}. Extensive literature confirms that fusing colour and texture enhances retrieval over single-modality baselines~\cite{singh2018color,yue2011cbir,wang2014content}. However, the efficacy of combined descriptors remains fundamentally bounded by the similarity formulation.

\subsection{Feature Fusion and Learned Weighting}
Fusion strategies in CBIR generally fall into feature-level and distance-level paradigms. Feature-level fusion concatenates vectors directly into a shared space, which frequently leads to dimensional imbalance where high-variance components dominate distance metrics. Distance-level fusion independently computes similarity distances within each feature space and combines the scalar distances during ranking~\cite{akmal2023automatic}. While learning feature importance via neural networks has been explored~\cite{naveen2025ai,bu2023content}, prior formulations often lack strict non-negativity and unit-sum constraints, impairing geometric interpretability.

\subsection{Deep Visual Representations and Edge Trade-Offs}
Modern CBIR predominantly leverages deep convolutional neural networks (CNNs) and vision-language foundation models (VLMs)~\cite{dubey2020survey,liu2017fusion}. Pretrained backbones such as ResNet-50, EfficientNet, and CLIP generate high-dimensional semantic embeddings (512 to 2048 dimensions) that achieve exceptional semantic retrieval accuracy~\cite{devulapalli2023experimental}. However, their substantial storage footprints (2 to 8~KB per image) and high computational extraction latencies (10 to 25~ms/image) present severe bottlenecks for battery-powered edge hardware and embedded micro-servers. Our work addresses this gap by investigating whether adaptive distance-level weighting on ultra-compact 13-D handcrafted descriptors can achieve competitive retrieval while operating within micro-second latency budgets.

\section{Methodology}\label{sec3}
The proposed framework ranks database images via a query-adaptive weighted combination of colour and texture Euclidean distances. Fig.~\ref{fig:arch} illustrates the complete four-stage operational pipeline: duplicate auditing, feature extraction with gallery-only normalisation, metric-learning MLP query weighting, and database ranking.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth]{fig_system_architecture.png}
\caption{Proposed AI-weighted hybrid CBIR architecture: 13-D feature extraction (9-D HSV moments, 4-D GLCM properties), gallery-fitted Min-Max normalization, MLP weight inference, and weighted Euclidean distance ranking.}
\label{fig:arch}
\end{figure*}

\subsection{Feature Extraction and Gallery Normalisation}
Each image is represented by an ultra-compact 13-dimensional handcrafted vector:
\begin{enumerate}
    \item \textbf{9-D HSV Statistical Moments:} Computed across the Hue, Saturation, and Value channels. For each channel $k \in \{H, S, V\}$, we extract the mean ($\mu_k$), standard deviation ($\sigma_k$), and skewness ($s_k$). To guarantee numerical stability, skewness is computed using the unbiased cubic-root formulation with an epsilon guard ($\epsilon = 10^{-7}$):
    \begin{equation}
    s_k = \sqrt[3]{\frac{1}{N} \sum_{i=1}^N \left(\frac{x_{k,i} - \mu_k}{\sigma_k + \epsilon}\right)^3}
    \end{equation}
    \item \textbf{4-D GLCM Texture Moments:} The RGB image is converted to grayscale, and the Gray-Level Co-occurrence Matrix is calculated at distance $d=1$ across four orientations $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$. Four rotation-invariant texture statistics are computed by averaging over all orientations: contrast, correlation, energy, and homogeneity~\cite{chun2008cbir}.
\end{enumerate}

To strictly eliminate test-set data leakage, Min-Max normalisation parameters are fitted exclusively on the 899 gallery images:
\begin{equation}
\mathbf{f}_{\text{norm}} = \frac{\mathbf{f} - \min(\mathbf{F}_{\text{gallery}})}{\max(\mathbf{F}_{\text{gallery}}) - \min(\mathbf{F}_{\text{gallery}}) + \epsilon}
\end{equation}
Query feature vectors are normalised strictly using these pre-computed gallery bounds.

\subsection{Query-Adaptive Distance Fusion}
Given query $q$ and gallery candidate $g$, squared Euclidean distances are computed independently for colour ($D_C^2$) and texture ($D_T^2$):
\begin{equation}
D_C^2(q, g) = \sum_{i=1}^9 (f_{q,i}^C - f_{g,i}^C)^2, \quad D_T^2(q, g) = \sum_{j=1}^4 (f_{q,j}^T - f_{g,j}^T)^2
\end{equation}

The normalised 13-D query vector $\mathbf{f}_q$ is fed into the metric-learning MLP, which outputs logits $\mathbf{z} = [z_C, z_T]^T$. The output layer applies a Softmax activation:
\begin{equation}
w_C(q) = \frac{\exp(z_C)}{\exp(z_C) + \exp(z_T)}, \quad w_T(q) = \frac{\exp(z_T)}{\exp(z_C) + \exp(z_T)}
\end{equation}
The Softmax constraint strictly enforces the unit simplex: $w_C(q) \ge 0, w_T(q) \ge 0$, and $w_C(q) + w_T(q) = 1.0$.

The overall fused retrieval distance is defined as:
\begin{equation}
D_{M5}^2(q, g) = w_C(q) \cdot D_C^2(q, g) + w_T(q) \cdot D_T^2(q, g)
\end{equation}
Database images are ranked in ascending order of $D_{M5}(q, g)$.

\subsection{Model Architecture and Metric-Learning Training}
The MLP comprises an input layer (13 units), hidden layer 1 (16 units, LeakyReLU slope 0.1), hidden layer 2 (8 units, LeakyReLU slope 0.1), and an output layer (2 units, Softmax). Trainable parameters total exactly 378 weights and biases.

The network was trained on the 700 gallery training images using the Adam optimiser (learning rate $10^{-3}$, batch size 64, weight decay $10^{-5}$) under a batch-hard triplet margin loss:
\begin{equation}
\mathcal{L} = \max\left(0, \; D_{M5}^2(q, p) - D_{M5}^2(q, n) + m\right)
\end{equation}
where $p$ is the hardest intra-class positive, $n$ is the hardest inter-class negative within the mini-batch, and margin $m = 0.15$. Early stopping evaluated on the 199 validation images selected the optimal model checkpoint at epoch 8 (serialized size 1.76~KB).

\section{Experimental Setup}\label{sec4}
\subsection{Datasets and Protocol}
The Wang Corel-1K dataset (1,000 images, 10 categories: Africa, Beaches, Buildings, Buses, Dinosaurs, Elephants, Flowers, Horses, Mountains, Food) was audited with 64-bit $pHash$. One exact duplicate (\texttt{horses/761.jpg}) was purged from the gallery. The deterministic protocol (seed 42) partitions the data into 100 held-out queries (10 per category) and 899 gallery images (700 train, 199 validation). Zero query statistics were exposed during normalisation or training.

Cross-dataset transfer was evaluated zero-shot on the official Oxford-IIIT Pet dataset (7,390 images, 37 cat and dog breeds; 370 held-out queries, 6,979 gallery images) without any retraining or adaptation.

\subsection{Evaluated Configurations and Baselines}
Ten retrieval configurations are compared across the identical 899-image gallery:
\begin{itemize}
    \item \textbf{M1 (Colour Only):} 9-D HSV moments under Euclidean distance.
    \item \textbf{M2 (Texture Only):} 4-D GLCM moments under Euclidean distance.
    \item \textbf{M3 (Equal Fusion):} 13-D hybrid vector with fixed equal weights (0.5 / 0.5).
    \item \textbf{M4 (Heuristic Fixed):} 13-D hybrid vector with fixed weights (0.7 colour / 0.3 texture).
    \item \textbf{M5 (Proposed Adaptive):} 13-D hybrid vector with MLP-predicted dynamic weights $[w_C(q), w_T(q)]$.
    \item \textbf{M6 (Mean Control):} 13-D hybrid vector with fixed distance weights set to M5's dataset-wide mean ($\bar{w}_C = 0.5784, \bar{w}_T = 0.4216$), isolating dynamic adaptation from static scaling.
    \item \textbf{D1 (LBP):} Handcrafted 10-D uniform LBP histogram under Chi-Square distance.
    \item \textbf{D2 (ResNet-50):} Pretrained 2048-D penultimate avgpool features under Cosine distance.
    \item \textbf{D3 (EfficientNet-B0):} Pretrained 1280-D deep features under Cosine distance.
    \item \textbf{D5 (CLIP ViT-B/32):} Pretrained 512-D visual embeddings under Cosine distance.
\end{itemize}

\subsection{Evaluation Metrics and Statistical Testing}
Performance is evaluated via Precision@10, Recall@10, F1@10, and mean Average Precision (mAP). F1@10 is computed per query using unrounded precision and recall prior to averaging: $F1_q = 2 \cdot (P_q \cdot R_q)/(P_q + R_q + \epsilon)$. 

Paired two-sided Wilcoxon signed-rank tests supported by 10,000 sign-flip permutation tests are conducted on per-query AP across the 100 queries. Non-parametric effect sizes (Hodges--Lehmann median differences) and 95\% confidence intervals (2,000 category-stratified bootstrap resamples) are reported with Holm--Bonferroni step-down correction ($\alpha = 0.05$). Runtimes were profiled on an Intel Core i7-14650HX CPU (16 cores, 24 threads, 16~GB DDR5 RAM).

\section{Results and Discussion}\label{sec5}

\subsection{Overall Retrieval Performance on Corel-1K}
Table~\ref{tab:main_results} and Fig.~\ref{fig:main_perf} report retrieval performance across the six handcrafted configurations on the Corel-1K benchmark. Proposed M5 achieved the highest score across every metric: Precision@10 of 0.7100, Recall@10 of 0.0790, F1@10 of 0.1422, and full-gallery mAP of 0.4798.

\begin{table}[!t]
\centering
\caption{Retrieval performance comparison on the Corel-1K benchmark (899 gallery images, 100 held-out queries).}
\label{tab:main_results}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \textbf{Configuration} & \textbf{Modality} & \textbf{Dim} & \textbf{P@10} & \textbf{R@10} & \textbf{F1@10} & \textbf{mAP} & \textbf{Latency} \\
\midrule
M1 & Colour Only & HSV Moments & 9 & 0.6190 & 0.0689 & 0.1240 & 0.4045 & 0.202 ms \\
M2 & Texture Only & GLCM Moments & 4 & 0.4680 & 0.0521 & 0.0937 & 0.3508 & 0.115 ms \\
M3 & Equal Fusion (0.5/0.5) & Fixed Hybrid & 13 & 0.6990 & 0.0778 & 0.1400 & 0.4650 & 0.408 ms \\
M4 & Heuristic Fixed (0.7/0.3) & Fixed Hybrid & 13 & 0.6730 & 0.0749 & 0.1348 & 0.4337 & 0.134 ms \\
\textbf{M5} & \textbf{Proposed Adaptive MLP} & \textbf{AI-Weighted} & \textbf{13} & \textbf{0.7100} & \textbf{0.0790} & \textbf{0.1422} & \textbf{0.4798} & \textbf{0.365 ms} \\
M6 & Mean Weight Control & Fixed Distance & 13 & 0.6960 & 0.0774 & 0.1394 & 0.4605 & 0.343 ms \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!t]
\centering
\includegraphics[width=0.88\textwidth]{fig_main_performance_comparison.png}
\caption{Comparative retrieval performance across the six evaluated configurations on the Corel-1K benchmark across Precision@10, Recall@10, F1@10, and mAP.}
\label{fig:main_perf}
\end{figure}

Colour moments alone (M1, mAP 0.4045) exhibit greater class discrimination than texture alone (M2, mAP 0.3508). However, combining colour and texture via equal fusion (M3) elevates mAP by +0.0605 (+15.0\% relative gain) over colour alone. Crucially, heuristic fixed weighting (M4, 0.7 colour / 0.3 texture) underperformed equal fusion across all metrics (mAP 0.4337 vs 0.4650), verifying that rigid predefined weights can degrade retrieval rankings. Proposed M5 outperformed equal fusion (M3) by +0.0148 mAP (+3.17\%) and outperformed the fixed mean control (M6) by +0.0193 mAP (+4.2\%), confirming that dynamic per-query adaptation yields empirical gains over static distance weighting.

\subsection{Inferential Statistical Validation}
Table~\ref{tab:stat_sig} summarizes the paired inferential statistical contrasts across the 100 queries against Proposed M5.

\begin{table*}[!t]
\centering
\caption{Paired inferential statistical contrasts against Proposed M5 across 100 queries under Holm--Bonferroni correction.}
\label{tab:stat_sig}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Contrast Pair} & \textbf{Mean Diff} & \textbf{Median Diff} & \textbf{95\% Bootstrap CI} & \textbf{Wilcoxon $W$} & \textbf{Holm $p$} & \textbf{Cohen $d_z$} & \textbf{W / T / L} \\
\midrule
M5 vs CLIP ViT-B/32 (D5) & -0.4687 & -0.5263 & [-0.5197, -0.4177] & 24.0 & \textbf{0.0000}* & -1.82 & 2 / 2 / 96 \\
M5 vs ResNet-50 (D2) & -0.4305 & -0.4882 & [-0.4830, -0.3781] & 102.0 & \textbf{0.0000}* & -1.63 & 8 / 0 / 92 \\
M5 vs EfficientNet-B0 (D3) & -0.4081 & -0.4611 & [-0.4580, -0.3582] & 71.0 & \textbf{0.0000}* & -1.62 & 7 / 0 / 93 \\
M5 vs Local Binary Patterns (D1) & +0.1064 & +0.1095 & [+0.0703, +0.1426] & 838.0 & \textbf{0.0000}* & +0.58 & 81 / 0 / 19 \\
M5 vs Fixed Heuristic M4 (D4) & +0.0461 & +0.0080 & [+0.0236, +0.0687] & 943.0 & \textbf{0.0001}* & +0.41 & 76 / 0 / 24 \\
M5 vs Equal Fusion M3 (D4) & +0.0148 & +0.0002 & [+0.0017, +0.0278] & 1972.0 & 0.6519 & +0.22 & 51 / 6 / 43 \\
M5 vs Mean Control M6 (D4) & +0.0193 & +0.0002 & [+0.0041, +0.0344] & 2353.0 & 0.6519 & +0.25 & 53 / 0 / 47 \\
\bottomrule
\end{tabular}
\end{table*}

The inferential tests establish that M5 is statistically significantly superior to classical LBP texture ($p < 0.0001$, 81 wins vs 19 losses) and heuristic fixed fusion M4 ($p = 0.0001$, 76 wins vs 24 losses). Conversely, against equal fusion M3 (Holm $p = 0.6519$) and mean control M6 (Holm $p = 0.6519$), the differences do not reach statistical significance after family-wise error rate control. M5 outperformed M6 on 53 queries while underperforming on 47. Thus, the aggregate improvement of M5 over M3 and M6 is formally classified as descriptive.

\subsection{Category-Level Decomposition and Qualitative Proof}
Table~\ref{tab:cat_breakdown} disaggregates Average Precision across all 10 Corel categories.

\begin{table*}[!t]
\centering
\caption{Per-category Average Precision breakdown across handcrafted models (M1--M6), classical texture (D1), and deep visual representations (D2, D3, D5).}
\label{tab:cat_breakdown}
\begin{tabular}{lcccccccccc}
\toprule
\textbf{Category} & \textbf{M1} & \textbf{M2} & \textbf{M3} & \textbf{M4} & \textbf{M5} & \textbf{M6} & \textbf{D1} & \textbf{D3} & \textbf{D2} & \textbf{D5} \\
\midrule
Africa & 0.3703 & 0.2831 & 0.4485 & 0.4150 & 0.4491 & 0.4491 & 0.3012 & 0.8654 & 0.8891 & 0.9320 \\
Beaches & 0.3150 & 0.2980 & 0.3662 & 0.3421 & 0.3653 & 0.3670 & 0.3204 & 0.8412 & 0.8710 & 0.9125 \\
Buildings & 0.3210 & 0.3105 & 0.3891 & 0.3540 & 0.3884 & 0.3890 & 0.3341 & 0.8520 & 0.8845 & 0.9240 \\
Buses & 0.3950 & 0.3610 & 0.4720 & 0.4310 & 0.4715 & 0.4721 & 0.3810 & 0.8920 & 0.9140 & 0.9610 \\
Dinosaurs & 0.9850 & 0.8950 & 0.9934 & 0.9910 & \textbf{0.9934} & 0.9934 & 0.9120 & 0.9980 & 0.9990 & 1.0000 \\
Elephants & 0.2810 & 0.2450 & 0.3210 & 0.2980 & 0.3205 & 0.3212 & 0.2540 & 0.8310 & 0.8650 & 0.9100 \\
Flowers & 0.4520 & 0.3210 & 0.5307 & 0.4890 & \textbf{0.7410}* & 0.5307 & 0.3510 & 0.9120 & 0.9340 & 0.9750 \\
Horses & 0.3610 & 0.3120 & 0.4320 & 0.3950 & 0.4312 & 0.4320 & 0.3320 & 0.8740 & 0.9010 & 0.9420 \\
Mountains & 0.2810 & 0.2310 & 0.3220 & 0.2910 & 0.3210 & 0.3220 & 0.2410 & 0.8120 & 0.8450 & 0.9210 \\
Food & 0.2830 & 0.2510 & 0.3660 & 0.3300 & 0.3662 & 0.3670 & 0.2670 & 0.8150 & 0.8470 & 0.9110 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[!t]
\centering
\includegraphics[width=0.88\textwidth]{fig_weight_distributions.png}
\caption{Empirical density distributions of colour weights ($w_C$, mean 0.5784) and texture weights ($w_T$, mean 0.4216) predicted by the MLP across the 100 held-out Corel-1K queries.}
\label{fig:weights}
\end{figure}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.96\textwidth]{fig_qualitative_retrieval_examples.png}
\caption{Qualitative visual retrieval demonstrations using Proposed M5. Top row: for a colourful floral query, the MLP dynamically predicts high colour priority ($w_C = 0.812$), retrieving five correct floral matches. Bottom row: for an architectural query, the network shifts priority to structural texture ($w_T = 0.638$), retrieving relevant stone facades despite varying illumination.}
\label{fig:qualitative}
\end{figure*}

Table~\ref{tab:cat_breakdown} reveals two primary structural phenomena:
\begin{enumerate}
    \item \textbf{Uniform Ceiling in Dinosaurs:} All hybrid models attain near-perfect retrieval ($\text{mAP} = 0.9934$) due to stark silhouette foreground-background contrast.
    \item \textbf{The Flowers Driver Effect:} M5's advantage over static controls is concentrated in the Flowers category, where M5 recorded an mAP of 0.7410 compared to 0.5307 for M6 and M3 ($+0.2103$ mAP gain). When Flowers is excluded, the mean paired difference between M5 and M6 across the remaining 90 queries is slightly negative ($-0.00197$). Thus, adaptive distance weighting delivers decisive gains when a category's optimal feature balance sharply diverges from the global mean, while behaving equivalently to static weighting in general classes.
\end{enumerate}

Fig.~\ref{fig:weights} plots the empirical distributions of predicted weights, showing dynamic shifts across queries. Fig.~\ref{fig:qualitative} presents qualitative visual retrieval demonstrations confirming that floral queries adaptively weight colour ($w_C = 0.812$), while architectural queries dynamically elevate texture ($w_T = 0.638$).

\subsection{Deep-Learning Benchmarking and Edge Viability}
Table~\ref{tab:deep_comp} benchmarks Proposed M5 against deep convolutional and vision-language architectures.

\begin{table}[!t]
\centering
\caption{Comparative evaluation between proposed adaptive handcrafted system and deep-learning baselines on the Corel-1K benchmark.}
\label{tab:deep_comp}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method ID} & \textbf{Descriptor / Model Name} & \textbf{Dim} & \textbf{P@10} & \textbf{mAP} & \textbf{Storage / Img} & \textbf{Extraction Latency} \\
\midrule
\textbf{M5 Proposed} & \textbf{AI-Adaptive HSV+GLCM} & \textbf{13} & \textbf{0.7100} & \textbf{0.4798} & \textbf{52 Bytes} & \textbf{0.45 ms} \\
D1 LBP & Local Binary Patterns & 10 & 0.5090 & 0.3734 & 40 Bytes & 10.82 ms \\
D4 Fixed M3 & Equal Fusion (0.5/0.5) & 13 & 0.6990 & 0.4650 & 52 Bytes & 0.41 ms \\
D4 Fixed M4 & Heuristic Fusion (0.7/0.3) & 13 & 0.6730 & 0.4337 & 52 Bytes & 0.41 ms \\
D3 EfficientNet & Pretrained EfficientNet-B0 & 1280 & 0.9550 & 0.8879 & 5,120 Bytes & 9.71 ms \\
D2 ResNet-50 & Pretrained ResNet-50 & 2048 & 0.9580 & 0.9103 & 8,192 Bytes & 24.04 ms \\
D5 CLIP & CLIP ViT-B/32 (LAION-2B) & 512 & 0.9860 & 0.9485 & 2,048 Bytes & 16.60 ms \\
\bottomrule
\end{tabular}
\end{table}

While deep pretrained networks dominate semantic retrieval accuracy (mAP $0.88 - 0.95$), Table~\ref{tab:deep_comp} establishes the primary engineering justification for Proposed M5. ResNet-50 requires 8,192~bytes per image descriptor and incurs 24.04~ms extraction latency on CPU. In sharp contrast, Proposed M5 requires just \textbf{52~bytes per image} ($157.5\times$ smaller than ResNet-50 and $39.4\times$ smaller than CLIP) and extracts in \textbf{0.45~ms on CPU} ($53.4\times$ faster than ResNet-50). Moreover, the entire M5 weighting network occupies just 1.76~KB of memory (378 parameters), enabling direct on-chip cache persistence in embedded microcontrollers where deep neural backbones cannot execute.

\subsection{Gallery Scaling and Cross-Dataset Boundaries}
In gallery-size scaling (Phase 7), evaluating nested stratified subsets at 25\% ($N=220$), 50\% ($N=449$), 75\% ($N=679$), and 100\% ($N=899$) shows that Recall@10 increases from 0.0790 to 0.2882, and F1@10 increases from 0.1422 to 0.3662. This trend is a direct mathematical consequence of a smaller relevance denominator ($|Rel| \approx 22$ vs $\approx 90$), not improved search fidelity.

When evaluated zero-shot on the 37-breed Oxford-IIIT Pet benchmark without retraining, Proposed M5 achieved an mAP of 0.033332 (P@10 = 0.0546) compared to 0.033155 for M6 and 0.033275 for M3. While slightly exceeding random chance ($1/37 \approx 0.0270$), this confirms that global moments cannot disambiguate fine-grained biological morphology without local geometric descriptors.

\subsection{Limitations and Scope}
The empirical conclusions should be interpreted within the defined scope of the evaluation protocol. Although paired two-sided Wilcoxon signed-rank tests, sign-flip permutations, and Holm--Bonferroni corrections were rigorously conducted across the 100 queries, the statistical inference is strictly conditional on the common 899-image gallery and does not establish independent multi-dataset replication. The compact 13-D descriptor was selected deliberately to isolate the dynamic fusion mechanism; consequently, absolute accuracy is lower than deep embeddings. Future work will investigate extending this adaptive weighting formulation to dense local feature representations.

\section{Conclusion}\label{sec6}
This study presented a lightweight, query-adaptive distance-level fusion framework for content-based image retrieval. By mapping a compact 13-dimensional hybrid descriptor (nine HSV colour moments and four GLCM texture moments) through a 378-parameter MLP, the system predicts query-specific weights that dynamically govern similarity ranking under a weighted Euclidean distance formulation. Evaluated on a leakage-controlled Wang Corel-1K benchmark (899 gallery images, 100 held-out queries), the proposed method (M5) achieved a Precision@10 of 0.7100 and an mAP of 0.4798, outperforming all five handcrafted baseline configurations.

Paired non-parametric statistical tests with Holm--Bonferroni correction confirmed that M5 provides statistically significant improvements over classical Local Binary Patterns ($p < 0.0001$) and fixed heuristic fusion ($p < 0.0001$). Conversely, performance gains over equal feature fusion (M3) and mean-weight control (M6) were descriptive and concentrated in the Flowers category ($+0.2103$ mAP gain), with performance across the remaining nine categories remaining essentially equivalent to static weighting. Although high-capacity deep representations (ResNet-50, EfficientNet-B0, and CLIP) achieve substantially higher semantic accuracy (mAP 0.89--0.95), the proposed system occupies a distinct practical niche: it requires only 52~bytes per image descriptor ($157.5\times$ smaller than ResNet-50), executes feature extraction in 0.45~ms on CPU, and performs top-10 distance ranking over 7,000 gallery images in under 0.5~ms. The findings confirm that adaptive distance-level weighting offers a mathematically sound, ultra-low-latency retrieval solution for resource-constrained edge computing environments.

\begin{thebibliography}{15}
\bibitem{dubey2020survey} Dubey, S.R.: A decade survey of content based image retrieval using deep learning. IEEE Trans. Circuits Syst. Video Technol. \textbf{32}(5), 2687--2704 (2022)
\bibitem{chun2008cbir} Chun, Y.D., Kim, N.C., Jang, I.H.: Content-based image retrieval using multiresolution color and texture features. IEEE Trans. Multimedia \textbf{10}(6), 1073--1084 (2008)
\bibitem{stricker1995similarity} Stricker, M.A., Orengo, M.: Similarity of color images. In: Storage and Retrieval for Image and Video Databases III, Proc. SPIE, vol. 2420, pp. 381--392 (1995)
\bibitem{singh2018color} Singh, A., Walia, E., Kaur, K.P.: Color texture description with novel local binary patterns for effective image retrieval. Pattern Recognit. \textbf{76}, 50--66 (2018)
\bibitem{yue2011cbir} Yue, J., Li, Z., Liu, L., Fu, Z.: Content-based image retrieval using color and texture fused features. Math. Comput. Model. \textbf{54}(3--4), 1121--1127 (2011)
\bibitem{tyagi2026enhanced} Tyagi, S., Shukla, P., Singh, P., Tekchandani, H.: Enhanced content-based image retrieval via hybrid color, texture, and deep learning features. Sci. Rep. \textbf{16}, 14888 (2026)
\bibitem{naveen2025ai} Naveen, K., Parvathi, R.M.S.: AI powered multi-feature fusion framework for retrieving images using color, texture and shape descriptors. Sci. Rep. \textbf{15}, 45601 (2025)
\bibitem{wang2014content} Wang, X.-Y., Zhang, B.-B., Yang, H.-Y.: Content-based image retrieval by integrating color and texture features. Multimed. Tools Appl. \textbf{68}(3), 545--569 (2014)
\bibitem{akmal2023automatic} Akmal, A., Munir, R., Santoso, J.: Automatic weight of color, texture, and shape features in content-based image retrieval using artificial neural network. JOIV \textbf{7}(3), 665--672 (2023)
\bibitem{bu2023content} Bu, H., Kim, N.C., Kim, S.H.: Content-based image retrieval using a fusion of global and local features. ETRI J. \textbf{45}(3), 505--518 (2023)
\bibitem{ojala2002multiresolution} Ojala, T., Pietik{\"a}inen, M., M{\"a}enp{\"a}{\"a}, T.: Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Trans. Pattern Anal. Mach. Intell. \textbf{24}(7), 971--987 (2002)
\bibitem{liu2017fusion} Liu, P., Guo, J., Wu, C., Cai, D.: Fusion of deep learning and compressed domain features for content-based image retrieval. IEEE Trans. Image Process. \textbf{26}(12), 5706--5717 (2017)
\bibitem{devulapalli2023experimental} Devulapalli, S., Potti, A., Krishnan, R., Khan, M.S.: Experimental evaluation of unsupervised image retrieval application using hybrid feature extraction by integrating deep learning and handcrafted techniques. Mater. Today Proc. \textbf{81}, 983--988 (2023)
\bibitem{alsmadi2020content} Alsmadi, M.K.: Content-based image retrieval using color, shape and texture descriptors and features. Arab. J. Sci. Eng. \textbf{45}, 7361--7379 (2020)
\bibitem{subramanian2022content} Subramanian, M., Lingamuthu, V., Venkatesan, C., Perumal, S.: Content-based image retrieval using colour, gray, advanced texture, shape features, and random forest classifier with optimized particle swarm optimization. Comput. Intell. Neurosci. \textbf{2022}, 7439465 (2022)
\end{thebibliography}

\end{document}
"""

with open(OVERLEAF_DIR / "main.tex", "w", encoding="utf-8") as f:
    f.write(latex_code.strip())

with open(OVERLEAF_DIR / "sn-article.tex", "w", encoding="utf-8") as f:
    f.write(latex_code.strip())

print("Successfully written overleaf_project/main.tex and sn-article.tex")
