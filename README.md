# Neuronal Cell-Type Classifier from MEA Electrophysiology

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

Classifying excitatory vs inhibitory neurons — and their subtypes — from multi-electrode array (MEA) electrophysiology recordings using machine learning and deep learning. Built by a neuroscientist who spent years doing this classification by hand.

---

## Motivation

Anyone who has worked with MEA recordings knows the process: you stare at raster plots and firing patterns, you compute ISI distributions, you look at spike shapes, and eventually you decide "this is an inhibitory interneuron" or "this is a pyramidal cell". It takes experience, it takes time, and two researchers don't always agree.

This project asks a simple question: **can a model learn the features that experienced electrophysiologists use intuitively, and how does its confusion matrix compare to human-level classification?**

The answer has practical implications for high-throughput MEA screens in drug discovery and disease modeling — including the iPSC-derived neuron platforms increasingly used in Parkinson's and epilepsy research.

---

## Dataset

**Source:** Allen Brain Cell Atlas — Electrophysiology dataset  
**URL:** https://celltypes.brain-map.org/data  
**Access:** publicly available, no registration required

The Allen Cell Types Database provides:
- Patch-clamp electrophysiology recordings from mouse and human cortical neurons
- ~1,500 cells with full morphological and transcriptomic annotations
- Standardised stimulus protocols (long square, short square, ramp, noise)
- Expert-curated cell-type labels (excitatory/inhibitory + subtype)

**Alternative / supplementary dataset:** NeuroMorpho.Org (morphological features) for multi-modal extension.

---

## Extended Dataset — Qiu, Lignani et al. (2022, Science) — Hippocampal MEA & Patch-Clamp

**Source:** Qiu Y, O'Neill N, Bhatt DL, Bhatt DL, Bhatt DL, Lignani G et al. "On-demand cell-autonomous gene therapy for brain circuit disorders." *Science* (2022). doi:[10.1126/science.abq6656](https://doi.org/10.1126/science.abq6656)  
**Dataset:** UCL Research Data Repository — [rdr.ucl.ac.uk/articles/dataset/Raw_Data.../20867117](https://rdr.ucl.ac.uk/articles/dataset/Raw_Data_/20867117)  
**Access:** publicly available

This dataset contains:
- Raw MEA recordings from hippocampal neurons (epilepsy circuit model)
- Raw patch-clamp electrophysiology files (.abf format)
- EEG recordings
- MEA analysis performed using the MATLAB pipeline developed by Prof. Michela Chiappalone (UniGe) and Ilaria Colombi (IIT)

**Why this matters for this project:** the network chain connecting this dataset to the primary author of this repository is direct. Gabriele Lignani (corresponding author, UCL) completed his PhD at the Italian Institute of Technology (IIT, Genova) — the same institution where I worked in the Benfenati lab — and is a co-author on my eLIFE 2022 paper. The MEA analysis pipeline used here (Chiappalone/Colombi, UniGe/IIT) is the same one used in the Maccione/Benfenati group where my MEA work was trained. The patch-clamp .abf files from hippocampal neurons in this dataset are directly compatible with the feature extraction pipeline implemented in this repository and could be re-analysed to extract E/I classification features, providing an independent validation set from a related but distinct experimental context.

**Why this dataset:** same experimental paradigm (patch-clamp single-cell recordings) as the work underlying several of my publications. More specifically, the Allen database includes hippocampal and cortical neurons recorded with the same stimulation protocols (long square, ramp, noise injection) that were standard in the Baldelli lab at the University of Genova and at IIT, where I characterised excitatory/inhibitory synaptic transmission and intrinsic excitability for over three years. The classification task this model attempts — distinguishing pyramidal cells from interneurons based on firing pattern features — is exactly what I was doing manually from HEKA/Igor traces during that period. The Allen dataset provides a ground truth against which to benchmark the model; my own experimental background provides the domain intuition to interpret where it fails.

---

## Feature Engineering

Electrophysiological features extracted per cell from raw traces:

| Feature group | Features |
|---|---|
| Firing rate | Mean firing rate, max instantaneous rate, adaptation index |
| Spike shape | AP threshold, peak amplitude, half-width, AHP depth, rise/decay time |
| Burst dynamics | Burst frequency, intra-burst frequency, burst duration |
| ISI statistics | Mean ISI, CV of ISI, ISI ratio (adaptation), log-ISI distribution |
| Subthreshold | Input resistance, membrane time constant, sag ratio |

Feature extraction implemented in Python using custom signal processing routines (scipy, numpy) — no black-box toolboxes.

---

## Models

### Baseline
- **Logistic Regression** (scikit-learn, multinomial) on hand-crafted features
- **Random Forest** on same feature set — interpretable, shows feature importance

### Main model
**MLP (PyTorch)**
```
Input (N features) → BatchNorm
  → Linear(N → 256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256 → 64) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(64 → K classes) → Softmax
```

### Extension (optional)
**1D-CNN on raw spike waveforms** — learns shape features directly from signal without manual feature engineering. Tests whether raw signal contains discriminative information beyond engineered features.

---

## Results

*To be updated after full cross-validation run.*

| Model | Task | Macro-F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | Exc vs Inh | — | — |
| Random Forest | Exc vs Inh | — | — |
| MLP | Exc vs Inh | — | — |
| MLP | Subtype (6-class) | — | — |
| 1D-CNN (raw spikes) | Exc vs Inh | — | — |

Validation: stratified 5-fold cross-validation. Metric of choice: macro-F1 (preferred over accuracy given class imbalance across subtypes).

---

## Project Structure

```
neuronal-classifier-mea/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/              # gitignored — download via script
│   └── processed/        # gitignored
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_mlp_pytorch.ipynb
│   └── 05_results_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py           # download + preprocessing
│   ├── features.py       # electrophysiology feature extraction
│   ├── model.py          # nn.Module classes
│   ├── train.py          # training loop + early stopping
│   └── evaluate.py       # metrics, confusion matrix, plots
├── configs/
│   └── mlp_default.yaml
├── results/
│   ├── confusion_matrices/
│   ├── training_curves/
│   └── metrics_summary.csv
└── scripts/
    ├── download_allen_data.py
    └── run_pipeline.py
```

---

## Installation

```bash
git clone https://github.com/sharkolsdj/neuronal-classifier-mea.git
cd neuronal-classifier-mea
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Download and preprocess data
python scripts/download_allen_data.py

# Run full pipeline (baseline + MLP)
python scripts/run_pipeline.py --config configs/mlp_default.yaml

# Or step by step via notebooks
jupyter lab notebooks/
```

---

## Key Dependencies

```
torch>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
allensdk>=2.16       # Allen Brain Institute SDK
jupyter>=1.0
```

---

## Author

**Daniele Ferrante, Ph.D.**  
Neuroscientist | Applied AI Developer  
[LinkedIn](https://www.linkedin.com/in/daniele-ferrante-60bb8381/) | [GitHub](https://github.com/sharkolsdj)

*Background note: the feature engineering in this project draws directly on 10 years of hands-on patch-clamp and MEA electrophysiology across three European research institutions — in particular the period at the University of Genova (Baldelli lab) and the Italian Institute of Technology (Benfenati lab, Genova), where I characterised the electrophysiology of excitatory and inhibitory hippocampal neurons and their synaptic properties in PRRT2-deficient, SynIIKO, and REST/NRSF-manipulated preparations. The feature set used here (ISI statistics, spike shape parameters, AP threshold, adaptation index) is built from the same conceptual vocabulary I used to interpret raw HEKA traces during that period. The model's confusion matrix is benchmarked against the same intuitive classification I performed manually for years.*

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Related Work

- Gouwens et al. (2019). Classification of electrophysiological and morphological neuron types in the mouse visual cortex. *Nature Neuroscience*.
- Scala et al. (2021). Phenotypic variation of transcriptomic cell types in mouse motor cortex. *Nature*.
- Ferrante D et al. (2021). PRRT2 modulates presynaptic Ca²⁺ influx by interacting with P/Q-type channels. *Cell Reports*. doi:10.1016/j.celrep.2021.109248
- Ferrea E, Maccione A, Medrihan L, Nieus T, Ghezzi D, Baldelli P, Benfenati F, Berdondini L (2012). Large-scale, high-resolution electrophysiological imaging of field potentials in brain slices with microelectronic multielectrode arrays. *Frontiers in Neural Circuits*. doi:10.3389/fncir.2012.00080
- Prestigio C\*, Ferrante D\* et al. (2021). REST/NRSF drives homeostatic plasticity of inhibitory synapses in a target-dependent fashion. *eLife*. doi:10.7554/eLife.69058
