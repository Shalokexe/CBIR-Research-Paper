# Phase 2 Feature Extraction and Normalization Validation Report

## 1. Feature Representation Summary
- **Total Images Processed**: 999 (899 Gallery + 100 Held-Out Queries)
- **Feature Vector Dimensionality**: Exactly 13 dimensions
  - **HSV Color Features**: 9 dimensions (Mean, Standard Deviation, Skewness for H, S, V)
  - **GLCM Texture Features**: 4 dimensions (Contrast, Correlation, Energy, Homogeneity averaged over 4 orientations)
- **Feature Integrity**:
  - Missing values: 0
  - Non-finite values (NaN / Inf): 0
  - Constant channel zero-variance handling: Fully protected ($\sigma < 10^{-6} \implies s = 0.0$)

## 2. Methodological Specifications
### HSV Color Feature Extraction
- **Color Conversion**: PIL standard RGB cast to `float32` and scaled to $[0.0, 1.0]$ via division by $255.0$. Converted to HSV via `matplotlib.colors.rgb_to_hsv`.
- **Value Ranges**: $H \in [0.0, 1.0)$, $S \in [0.0, 1.0]$, $V \in [0.0, 1.0]$.
- **Calculation Type**: Pixel-based calculation across flattened channel arrays.
- **Moments**:
  $$\mu = \frac{1}{N} \sum x_i, \quad \sigma = \sqrt{\frac{1}{N} \sum (x_i - \mu)^2}, \quad s = \frac{\frac{1}{N} \sum (x_i - \mu)^3}{\sigma^3}$$
- **Zero Variance Fallback**: When $\sigma < 10^{-6}$, skewness is assigned exactly $0.0$.

### GLCM Texture Feature Extraction
- **Grayscale Conversion**: PIL 8-bit grayscale mode (`'L'`) with values in $[0, 255]$.
- **Quantization Levels**: 256 gray levels (`levels=256`).
- **Pixel Distance**: 1 pixel (`distances=[1]`).
- **Orientations**: 4 directions: $0^\circ, 45^\circ, 90^\circ, 135^\circ$ ($0, \pi/4, \pi/2, 3\pi/4$ radians).
- **Matrix Settings**: `symmetric=True`, `normed=True`.
- **Directional Averaging**: Computed via scikit-image's `graycoprops` and averaged:
  $$\text{Property} = \frac{1}{4} \sum_{\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}} \text{graycoprops}(GLCM_\theta, \text{Property})$$

## 3. Gallery-Only Min-Max Normalization
- **Strict Leakage Prevention**: Parameters $\min$ and $\max$ were fitted exclusively on the 899 gallery images.
- **Fitted Bounds**:
  - **Color Min**: `[ 0.0809,  0.0407, -4.1536,  0.039 ,  0.082 , -2.3983,  0.1481,  0.101 ,
 -4.2217]`
  - **Color Max**: `[0.871 , 0.4574, 6.767 , 0.8728, 0.4078, 4.9858, 0.8693, 0.3893, 2.8565]`
  - **Texture Min**: `[1.8242e+01, 6.6281e-01, 8.1672e-03, 6.2057e-02]`
  - **Texture Max**: `[1.9878e+03, 9.9599e-01, 6.0952e-01, 8.4322e-01]`
- **Gallery Output Range**: All 899 gallery normalized values lie strictly within $[0.0, 1.0]$.
- **Query Set Isolation**: Held-out queries were transformed using the frozen gallery scaler without updating parameter state.

## 4. Extraction Timing and Performance
- **Total Extraction Duration**: 46.80 seconds
- **Mean Processing Time per Image**: 0.0468 seconds (46.85 ms/image)

## 5. Artifacts Generated
- `output/phase2/features.npz`: Compressed archive with arrays `raw_color`, `raw_texture`, `normalized_color`, `normalized_texture`.
- `output/phase2/features.csv`: Tabular CSV containing raw and normalized features for all 999 evaluated images.
- `output/phase2/feature_extraction_config.json`: Complete specification file enabling deterministic reproduction.
- `output/phase2/feature_metadata.json`: Serialized extraction metrics, timestamps, and normalization bounds.
- `figures/phase2_feature_distributions.png`: High-resolution boxplot of color and texture distributions.
- `figures/phase2_representative_descriptors.png`: Category-level descriptor signature comparisons.

## 6. Verification Status
- **Validation**: **PASSED**.
- All mathematical bounds, shapes, and zero-leakage constraints verified successfully.
