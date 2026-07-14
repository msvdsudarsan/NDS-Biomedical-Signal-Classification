# Data Availability

All datasets used in this study are publicly available via PhysioNet. No
proprietary or restricted data were used, and no synthetic data are used for the
reported results.

---

## 1. CHB-MIT Scalp EEG Database

- URL: https://physionet.org/content/chbmit/1.0.0/
- DOI: https://doi.org/10.13026/C2K01R
- Scope used here: subjects chb01, chb02, chb03, chb05; channel FP1-F7; 256 Hz;
  2-second windows; 400 ictal + 400 interictal windows (balanced).
- Access: free registration on PhysioNet.

## 2. MIT-BIH Arrhythmia Database

- URL: https://physionet.org/content/mitdb/1.0.0/
- DOI: https://doi.org/10.13026/C2F305
- Scope used here: canonical AAMI inter-patient split. DS1 = {101,106,108,109,
  112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230}
  (train); DS2 = {100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,
  221,222,228,231,232,233,234} (test). Paced records (102,104,107,217) excluded.
  Lead MLII; 360 Hz; five AAMI classes (N, S, V, F, Q).
- Access: freely available, no registration required.

---

## Reproducibility Note

The MATLAB script `NDS_RealData_Pipeline.m` downloads the real recordings above
from PhysioNet and computes every reported number directly from them, with no
hardcoded results and no synthetic substitution. Class balancing on the ECG task
is performed by random oversampling of the DS1 training partition only.

---

## Dataset Citations

```
Shoeb, A. H. (2009). Application of Machine Learning to Epileptic Seizure Onset
Detection and Treatment. PhD Thesis, MIT. CHB-MIT database, PhysioNet.

Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia
Database. IEEE Eng. Med. Biol. Mag., 20(3), 45-50. MIT-BIH database, PhysioNet.
```
