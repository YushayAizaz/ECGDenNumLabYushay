# ECG Baseline Wander Removal

**Research internship | ELTE Department of Numerical Analysis**

This repository presents the research code and experimental workflow for my internship at Eötvös Loránd University (ELTE), focused on benchmarking baseline wander removal from PhysioNet 2020 ECG signals. The project compares neural networks with classical numerical optimization to investigate the trade-off between reconstruction accuracy and computational cost.

Baseline wander is a slowly varying disturbance that shifts an ECG signal's baseline. The objective is to remove this interference while preserving the underlying cardiac waveform.

## Benchmarking scope

| Method | Role in the comparison |
| --- | --- |
| **CNN** | An end-to-end convolutional neural network for ECG denoising. |
| **FISTA-Net** | A learned iterative network connecting sparse approximation with neural network training. |
| **FISTA** | The Fast Iterative Shrinkage-Thresholding Algorithm, used as a classical sparse optimization baseline. |
| **QP** | A quadratic programming solution of basis pursuit denoising, used as an optimization reference. |

The workflow combines ECG and noise preparation, K-SVD dictionary learning, model training and inference, waveform inspection, reconstruction-error analysis, and runtime measurements. It supports comparison of data-driven denoising with dictionary-based numerical methods; no aggregate performance ranking is asserted in this summary.

## Repository guide

| Location | Contents |
| --- | --- |
| `initial_analysis.ipynb`, `generate_data.ipynb` | Signal exploration and experimental data preparation. |
| `KSVD_dictionary-learning.ipynb` | Learning a dictionary for sparse approximation. |
| `BPDN_QP-vs-FISTA.ipynb` | Comparison of quadratic programming and FISTA. |
| `FISTA-Net_pre-training.ipynb`, `FISTA-Net_transfer-learning.ipynb` | FISTA-Net pre-training and fine-tuning. |
| `FISTA-Net_inference.ipynb` | Inference with a local FISTA-Net checkpoint. |
| `analyze_trained_model_mlflow.ipynb`, `meas_runtimes.ipynb` | Model analysis and runtime benchmarking. |
| `src/` | CNN, FISTA-Net, K-SVD, and reference denoising implementations. |
| `models/` | Retained CNN and FISTA-Net checkpoints and their environment metadata. |
| `data/` | Smaller supporting data and the retained baseline-wander dictionary. |

## Data availability

**The full raw PhysioNet 2020 dataset files are omitted due to size.** This repository is a compact code and research-artifact release, not a complete dataset distribution. Smaller supporting data, including MIT-BIH noise records, remain where present.

Files larger than 20 MB (20,000,000 bytes) are excluded from this upload copy. This includes `data/ECG-QDB_cat-1.npy`; notebooks that depend on it require the data to be restored locally. Obtain the original datasets separately and adjust notebook data paths before running the experiments.

## Using the research code

Use a Python environment with Jupyter and the dependencies required by the selected notebook. The saved models include `requirements.txt`, `conda.yaml`, and `python_env.yaml` files documenting their environments. The workflow uses PyTorch, NumPy, Matplotlib, WFDB, MLflow, and optimization libraries such as CVXOPT and PyLops.

For training notebooks that use experiment tracking, start a local MLflow server:

```shell
mlflow server --host 127.0.0.1 --port 8080
```

The original training workflow uses 3,000 pre-training epochs followed by 17,000 fine-tuning epochs. Restart the notebook kernel between these stages when needed to release GPU memory. The retained FISTA-Net checkpoint is in `models/FISTA-Net_ep20000/`, and the dictionary is in `data/dictionary_BW_real_data.npy`.

Some analysis notebooks retain earlier import layouts, local data paths, and MLflow run references. These need to be aligned with the current folder structure and local environment before execution; this upload is not a self-contained, one-command reproduction package.

## Research context

The original repository README identifies this codebase with the paper **Sparse dictionary learning neural networks for ECG signal denoising**. This internship summary preserves that research context.
