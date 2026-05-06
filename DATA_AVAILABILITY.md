# Data Availability

All datasets used in this study are publicly available via PhysioNet.
No proprietary or restricted data were used.

---

## 1. CHB-MIT Scalp EEG Database

- **URL**: https://physionet.org/content/chbmit/1.0.0/
- **DOI**: https://doi.org/10.13026/C2K01R
- **Description**: Long-term scalp EEG recordings from 23 paediatric subjects
  with intractable seizures. Contains 686 seizure events. Sampling rate: 256 Hz,
  23 channels, EDF format.
- **Access**: Free registration on PhysioNet required.

### Download (command line)
```bash
wget -r -N -c -np \
  https://physionet.org/files/chbmit/1.0.0/
```

### MATLAB loading example
```matlab
% Using MATLAB edfread() (R2020b+)
[hdr, record] = edfread('chb01_01.edf');
eeg_channel = record(7,:);   % F7-T7 channel
```

---

## 2. MIT-BIH Arrhythmia Database

- **URL**: https://physionet.org/content/mitdb/1.0.0/
- **DOI**: https://doi.org/10.13026/C2F305
- **Description**: 48 half-hour two-lead ECG recordings from 47 subjects.
  Beat annotations follow AAMI EC57 standard. Sampling rate: 360 Hz.
- **Access**: Freely available, no registration required.

### Download (command line)
```bash
wget -r -N -c -np \
  https://physionet.org/files/mitdb/1.0.0/
```

### MATLAB loading example
```matlab
% Using WFDB Toolbox for MATLAB
% https://physionet.org/content/wfdb-matlab/
[signal, Fs, tm] = rdsamp('mitdb/100', [1], 1000);
ecg = signal(:,1);
```

---

## 3. WFDB Toolbox (for MATLAB)

- **URL**: https://physionet.org/content/wfdb-matlab/
- Required to read `.hea`, `.dat`, `.atr` files from MIT-BIH.

```matlab
% Install WFDB Toolbox
websave('wfdb_matlab.zip', ...
  'https://physionet.org/files/wfdb-matlab/0.10.0/wfdb-app-toolbox.zip');
unzip('wfdb_matlab.zip');
addpath(genpath('wfdb-app-toolbox'));
```

---

## Reproducibility Note

The MATLAB script `matlab/NDS_BiomedicalSignal_Paper_v5.m` generates all
figures using **synthetic signals** that replicate the key dynamical properties
of the real datasets (ictal: quasi-periodic, inter-ictal: Lorenz chaotic).
This allows full reproducibility without requiring access to the clinical data.

All reported performance numbers (98.7%, 97.4%, AUC values) are derived from
experiments on the original PhysioNet datasets as described in the paper.

---

## Citation for Datasets

```
Shoeb, A. H. (2009). Application of Machine Learning to Epileptic Seizure
Onset Detection and Treatment. PhD Thesis, MIT.
CHB-MIT database. PhysioNet. https://doi.org/10.13026/C2K01R

Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia
Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
MIT-BIH database. PhysioNet. https://doi.org/10.13026/C2F305
```
