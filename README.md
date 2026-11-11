# Schizophrenia detection from real EEG and MRI data

One script, two separate models. The EEG model and the MRI model never see each other's data, so they can use different people (for example Warsaw EEG and COBRE MRI).

```
pip install -r requirements.txt
python sz_real_data.py --data_dir data
```

## What the script does

1. Reads every file in `data/eeg/` and `data/mri/` and works out each subject's label (SZ = 1, HC = 0).
2. Turns each subject into one row of numbers.
   - EEG: relative power in five bands (delta, theta, alpha, beta, gamma) for every channel, from the whole recording. 19 channels give 95 numbers.
   - MRI: the correlation between every pair of brain regions (functional connectivity). 30 regions give 435 numbers.
3. Scores an SVM with 5-fold cross-validation over subjects, repeated 10 times with different shuffles. It prints accuracy, sensitivity, specificity and AUC as mean +/- standard deviation.
4. Gives a YES/NO answer for every subject ("schizophrenia?"), made while that person was hidden from training, and saves the table as `predictions_eeg.csv` and `predictions_mri.csv`.

Because there is one row per subject, no person is in both the training and the test set.

## YES / NO output

Every subject gets a row like this:

```
subject actual schizophrenia?  score correct
    s01     SZ            YES  0.597     yes
    h03     HC            YES  0.385      NO
```

`YES` means the model says schizophrenia. The score is the SVM's distance from its decision line. Above 0 is YES, below 0 is NO, and values near 0 are uncertain. It is not a probability.

To get an answer for one NEW recording that is not in your data folder:

```
python sz_real_data.py --data_dir data --predict_eeg new_person.edf
python sz_real_data.py --data_dir data --predict_mri new_person.csv
```

The model is trained on all labelled subjects first, then prints `Schizophrenia: YES` or `NO`. If the file is also in the training folder, the script warns you, because that answer is not a fair test.

These answers come from a small research model. They are not a diagnosis.

## Get the data

### EEG: Warsaw "EEG in schizophrenia" (14 patients, 14 controls)

Download all 28 files from RepOD (DOI 10.18150/repod.0107441) and put them in `data/eeg/`. They are EDF files named `h01.edf` to `h14.edf` (controls) and `s01.edf` to `s14.edf` (patients), recorded at 250 Hz with 19 channels. The script reads the labels from the `h` and `s` in the file names. No extra library is needed for EDF.

### MRI: COBRE (72 patients, 74 controls)

COBRE is on figshare as "COBRE preprocessed with NIAK 0.12.4". I have not checked the exact files inside that download, so look at what you got:

- If you have ROI time series as `.csv`, `.txt`, `.1D`, `.npy` or `.mat` (rows = time points, columns = regions), copy them into `data/mri/`. If the rows are regions instead, add `--roi_first`.
- If you have 4D NIfTI images (`.nii` or `.nii.gz`), copy them into `data/mri/` and run `pip install nilearn nibabel`. The script extracts 39 region time series per scan with the MSDL atlas. The atlas is downloaded the first time, so you need internet then.

For MRI you must provide labels. Create `data/mri/labels.csv`:

```
subject_id,label
0040000,Patient
0040001,Control
```

The first column must match the file name without its extension. Labels can be `Patient`/`Control`, `SZ`/`HC`, or `1`/`0`. You can also skip the CSV and sort the files into folders named `sz/` and `hc/`.

Final layout:

```
data/
  eeg/   h01.edf ... s14.edf
  mri/   labels.csv, one file per subject
```

You can run only one modality. If `data/mri/` does not exist, the script runs just the EEG model.

## Test before using real data

```
python make_example_data.py                  # writes fake files in the real layout
python sz_real_data.py --data_dir data_example
python test_loaders.py                       # checks the readers
python sz_real_data.py                       # fake in-memory data, checks the install
```

The example files contain made-up signals. Only their formats are realistic.

## Options

```
--model svm|lr|rf     classifier (default svm)
--repeats 10          how many times to repeat the cross-validation
--fs 250              sampling rate for non-EDF EEG files
--eeg_channels 19     keep only the first N channels
--roi_first           MRI files are regions x time
--predict_eeg FILE    YES/NO for one new EEG file
--predict_mri FILE    YES/NO for one new MRI file
```

## What has and has not been tested

Tested: the EDF reader (on EDF files written by `make_example_data.py`), the other file formats, the three ways of labelling, and the full run on fake data in the real layout.

Not tested: the real Warsaw and COBRE downloads, because they could not be fetched where this code was written. The NIfTI route was also not run, since nilearn was not installed there. If the EDF reader fails on a real file, the script prints the file name and the error. You can convert with MNE and keep going:

```python
import mne, numpy as np
raw = mne.io.read_raw_edf("s01.edf", preload=True)
np.save("s01.npy", raw.get_data())          # channels x samples
```

Then run with `--fs 250` and keep the file name (`s01.npy`).

## Reading the results

- The Warsaw set has 28 people. With so few, the AUC can move by 0.1 or more between shuffles. Look at the +/- value and do not quote a single run.
- There is no artifact removal and no filtering beyond using the 1 to 45 Hz range for the power calculation.
- Age, sex, medication and scanner are not controlled. Results are not clinical evidence.
- Good practice for the report: state the dataset sizes, say that validation is subject-level, and say that EEG and MRI came from different people.
