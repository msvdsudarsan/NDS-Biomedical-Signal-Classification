# Nonlinear Dynamical System-Based Feature Extraction for Biomedical Signal Classification Using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2026a-blue.svg)](https://www.mathworks.com)
[![Journal](https://img.shields.io/badge/Journal-BSPC%20(Elsevier)-orange.svg)](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control)

---

## Authors

**Sri Venkata Durga Sudarsan Madhyannapu**¹ · **Kankipati Subbarao**¹ · **G. Arunagiri**² · **S. Parvathi**³ · **A. Kiran Kumar**⁴ · **D. R. Krishna Thippisetti**⁵

¹ Department of Mathematics, School of Sciences, Humanities and Management, Dr. RVR NRI Institute of Technology (Deemed to be University), Pothavarappadu, Andhra Pradesh 521212, India · ORCID: Madhyannapu 0009-0001-2126-6428 (corresponding), Subbarao 0009-0000-3953-2950

² Department of Mathematics, Sri Indu Institute of Engineering and Technology, Sheriguda, Hyderabad, Telangana 501510, India · ORCID: Arunagiri 0009-0002-3657-8819

³ Department of Mathematics and Statistics, SRM Institute of Science and Technology, Kattankulathur, Chennai, Tamilnadu 603203, India

⁴ Department of Mathematics, S R K R Engineering College, Bhimavaram, Andhra Pradesh 534204, India · ORCID: Kiran Kumar 0009-0003-2097-7249

⁵ Department of Mathematics, Sri Vasavi Engineering College (Autonomous), Pedatadepalli, Tadepalligudem, Andhra Pradesh 534101, India

---

## Abstract

We present a biomedical signal classification framework that couples nonlinear
dynamical systems theory with deep learning. Raw EEG/ECG signals are denoised
by singular system analysis (SSA), reconstructed in phase space via the Takens
embedding theorem, and summarised by a compact ten-dimensional feature vector
comprising the largest Lyapunov exponent, the correlation dimension, the
embedding parameters, and six recurrence quantification analysis (RQA)
statistics. Two modality-appropriate supervised back-ends are evaluated on
real PhysioNet recordings: a hybrid CNN-LSTM network classifies the
SSA-denoised EEG waveform windows, while a radial-basis-function support
vector machine (one-vs-all ECOC) classifies the ECG feature vector augmented
with three RR-interval descriptors. **All numbers below are measured directly
by the pipeline on real data (no hardcoding).**

---

## Measured Results (real data, MATLAB R2026a)

### EEG seizure detection (CHB-MIT, subjects chb01/02/03/05)

| Protocol | Accuracy | Sens. | Spec. | AUC |
|----------|----------|-------|-------|-----|
| 10-fold stratified CV | 87.4% ± 2.6 | 89.8% | 85.0% | 0.945 |
| Leave-one-subject-out | 82.3% ± 2.7 | — | — | — |

### ECG arrhythmia classification (MIT-BIH, inter-patient AAMI DS1/DS2)

| Metric | Value |
|--------|-------|
| Overall accuracy | 62.95% |
| Macro-averaged AUC | 0.802 |

Per-class sensitivity: N 68.5%, S 52.2%, V 83.8%, F 47.9%, Q 0.0%.
The S-class sensitivity and the severely under-represented Q class are the
principal open problems and are discussed openly in the paper.

> These are honest single-source benchmark results on a small EEG cohort and the
> canonical inter-patient ECG split. They are reported as-is rather than as
> state-of-the-art claims.

---

## Reproducing the results

```matlab
% MATLAB R2026a (Deep Learning + Statistics toolboxes)
% Set CFG.quick_demo = false for the full run.
NDS_RealData_Pipeline
```

The script downloads the required CHB-MIT EDF files and MIT-BIH records directly
from PhysioNet on first run, extracts the dynamical features, trains the
modality-appropriate classifiers (CNN-LSTM for EEG; RBF-SVM for ECG), prints
the measured metrics, and saves all figures (PNG + PDF) to the output folder.
No synthetic data are used.

---

## Datasets

| Dataset | Task | Scope used | Source |
|---------|------|-----------|--------|
| CHB-MIT Scalp EEG | Seizure vs. non-seizure | subjects chb01/02/03/05 | PhysioNet |
| MIT-BIH Arrhythmia | 5-class AAMI (inter-patient) | DS1 train / DS2 test | PhysioNet |

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for full download instructions.

---

## Citation

```bibtex
@article{madhyannapu2026nonlinear,
  title   = {Nonlinear Dynamical System-Based Feature Extraction for
             Biomedical Signal Classification Using Deep Learning},
  author  = {Madhyannapu, Sri Venkata Durga Sudarsan and Subbarao, Kankipati and Arunagiri, G. and Parvathi, S. and Kiran Kumar, A. and Thippisetti, D. R. Krishna},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE). Datasets remain subject to their PhysioNet licenses.
