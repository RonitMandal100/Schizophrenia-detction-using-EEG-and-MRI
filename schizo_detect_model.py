
import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]  
STD_19 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
          "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
SZ_WORDS = {"1", "sz", "scz", "schizophrenia", "schizophrenic", "patient", "patients"}
HC_WORDS = {"0", "hc", "control", "controls", "healthy", "healthy control"}


# ======================================================================
# 1. READING FILES
# ======================================================================
def read_edf(path):
    
    with open(path, "rb") as f:
        head = f.read(256)
        n_rec, dur, ns = int(head[236:244]), float(head[244:252]), int(head[252:256])
        sig = f.read(256 * ns)
        raw = np.frombuffer(f.read(), dtype="<i2")

    def field(offset, width, i):                      # i-th signal's value of one header field
        s = offset * ns + i * width
        return sig[s:s + width].decode("ascii", "ignore").strip()

    labels = [field(0, 16, i) for i in range(ns)]
    pmin, pmax = [[float(field(o, 8, i)) for i in range(ns)] for o in (104, 112)]
    dmin, dmax = [[float(field(o, 8, i)) for i in range(ns)] for o in (120, 128)]
    nsamp = [int(field(216, 8, i)) for i in range(ns)]

    rec_len = sum(nsamp)
    if n_rec <= 0:                                    # -1 means "unknown"
        n_rec = len(raw) // rec_len
    raw = raw[: n_rec * rec_len].reshape(n_rec, rec_len)

    data, names, start = [], [], 0
    for i in range(ns):
        block = raw[:, start:start + nsamp[i]].reshape(-1).astype(float)
        start += nsamp[i]
        if "ANNOTATION" in labels[i].upper():
            continue
        scale = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        data.append((block - dmin[i]) * scale + pmin[i])
        names.append(labels[i])
    fs = nsamp[0] / dur
    return np.array(data), fs, names


def pick_standard_19(data, names):
    """Keep the 19 standard 10-20 channels (in a fixed order) if the file names them."""
    clean = [re.sub(r"(EEG|-REF|-LE|-AV|[\s_\-.])", "", n.upper()) for n in names]
    if all(c in clean for c in STD_19):
        return data[[clean.index(c) for c in STD_19]]
    return data


def read_matrix(path):
    """Read a numeric table (.npy, .mat, or text/csv/1D, header lines are ignored)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(float)
    if ext == ".mat":
        arrays = [v for k, v in loadmat(path).items() if not k.startswith("__")
                  and isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number)]
        return max(arrays, key=lambda a: a.size).astype(float)
    df = pd.read_csv(path, sep=r"[,;\s]+", engine="python", header=None, comment="#")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")
    return df.to_numpy(float)


def load_eeg(path, fs_default, n_channels):
    """-> (channels x samples, sampling rate)"""
    if path.lower().endswith(".edf"):
        data, fs, names = read_edf(path)
        data = pick_standard_19(data, names)
    else:
        data, fs = read_matrix(path), fs_default
        if data.shape[0] > data.shape[1]:             # saved as samples x channels
            data = data.T
    return (data[:n_channels] if n_channels else data), fs


_masker = {}


def load_mri(path, roi_first):
    """-> time x ROIs"""
    if path.lower().endswith((".nii", ".nii.gz")):    # 4D image -> ROI time series via an atlas
        if "m" not in _masker:
            from nilearn.datasets import fetch_atlas_msdl
            try:
                from nilearn.maskers import NiftiMapsMasker
            except ImportError:
                from nilearn.input_data import NiftiMapsMasker
            _masker["m"] = NiftiMapsMasker(fetch_atlas_msdl().maps, standardize="zscore_sample")
        return _masker["m"].fit_transform(path)
    ts = read_matrix(path)
    return ts.T if roi_first else ts


# ======================================================================
# 2. LABELS
# ======================================================================
def stem(path):
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def parse_label(value):
    v = str(value).strip().lower()
    return 1 if v in SZ_WORDS else 0 if v in HC_WORDS else None


def find_labelled_files(folder):
    """Label comes from (1) labels.csv, (2) a folder named sz/hc/patient/control,
    (3) a file name like s01 / h01 (Warsaw EEG)."""
    table = {}
    lp = os.path.join(folder, "labels.csv")
    if os.path.exists(lp):
        df = pd.read_csv(lp, dtype=str)
        table = {str(a).strip(): parse_label(b) for a, b in zip(df.iloc[:, 0], df.iloc[:, 1])}
    out, skipped = [], []
    for root, _, names in os.walk(folder):
        for n in sorted(names):
            if n.startswith(".") or n == "labels.csv":
                continue
            path = os.path.join(root, n)
            label = table.get(stem(path))
            if label is None:
                label = parse_label(os.path.basename(root))
            if label is None:
                m = re.match(r"^([sh])\d+$", stem(path).lower())
                label = None if not m else int(m.group(1) == "s")
            if label is None:
                skipped.append(path)
            else:
                out.append((path, label))
    if skipped:
        print(f"  skipped {len(skipped)} file(s) with no label, e.g. {os.path.basename(skipped[0])}")
    return sorted(out)


# ======================================================================
# 3. FEATURES (one row per subject)
# ======================================================================
def eeg_features(data, fs):
    """Relative power in 5 bands for each channel (whole recording, Welch average)."""
    f, psd = welch(data, fs=fs, nperseg=int(2 * fs), axis=-1)
    total = psd[:, (f >= 1) & (f <= 45)].sum(axis=1)
    return np.concatenate([psd[:, (f >= lo) & (f < hi)].sum(axis=1) / total for lo, hi in BANDS])


def mri_features(ts):
    """Functional connectivity: correlation between every pair of ROIs."""
    corr = np.nan_to_num(np.corrcoef(ts.T))
    return corr[np.triu_indices(corr.shape[0], k=1)]


# ======================================================================
# 4. MODEL + EVALUATION
# ======================================================================
def make_model(name):
    clf = {"svm": SVC(kernel="rbf"), "lr": LogisticRegression(C=0.1, max_iter=2000),
           "rf": RandomForestClassifier(n_estimators=300, random_state=0)}[name]
    return make_pipeline(StandardScaler(), clf)


def evaluate(X, y, names, title, model_name, repeats, out_csv):
    """Repeated stratified K-fold over subjects (one row = one subject, so nobody is in
    both train and test). Every subject is predicted while hidden from training.
    Repeating with new shuffles shows how unstable small data is."""
    n_splits = int(min(5, np.bincount(y).min()))
    rows, all_scores = [], []
    for seed in range(repeats):
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        if model_name == "rf":
            score = cross_val_predict(make_model("rf"), X, y, cv=cv, method="predict_proba")[:, 1] - 0.5
        else:
            score = cross_val_predict(make_model(model_name), X, y, cv=cv, method="decision_function")
        all_scores.append(score)
        tn, fp, fn, tp = confusion_matrix(y, (score > 0).astype(int)).ravel()
        rows.append([(tp + tn) / len(y), tp / (tp + fn), tn / (tn + fp), roc_auc_score(y, score)])
    r = np.array(rows)
    print(f"\n{title}: {len(y)} subjects (SZ {y.sum()}, HC {(1 - y).sum()}), {X.shape[1]} features, "
          f"model = {model_name}, {n_splits}-fold x {repeats}")
    for i, n in enumerate(["Accuracy   ", "Sensitivity", "Specificity", "AUC        "]):
        print(f"  {n}: {r[:, i].mean():.3f} +/- {r[:, i].std():.3f}")
    if len(y) < 40:
        print("  (fewer than 40 subjects: treat these numbers as very rough)")

    # yes/no for every subject: average score over the repeats, above 0 = schizophrenia
    mean_score = np.mean(all_scores, axis=0)
    pred = (mean_score > 0).astype(int)
    df = pd.DataFrame({"subject": names,
                       "actual": ["SZ" if v else "HC" for v in y],
                       "schizophrenia?": ["YES" if p else "NO" for p in pred],
                       "score": mean_score.round(3),
                       "correct": np.where(pred == y, "yes", "NO")})
    df.to_csv(out_csv, index=False)
    print(f"\n  Prediction for each subject (made while that subject was hidden from training):")
    if len(df) <= 40:
        print("  " + df.to_string(index=False).replace("\n", "\n  "))
    print(f"  {int((pred == y).sum())} of {len(y)} subjects correct. Full table saved to {out_csv}")


def predict_new(X, y, x_new, model_name, what, path):
    """Train on ALL labelled subjects, then say YES/NO for one new recording."""
    if len(x_new) != X.shape[1]:
        raise SystemExit(f"{os.path.basename(path)} gives {len(x_new)} features but the training data "
                         f"has {X.shape[1]}: different number of channels/ROIs.")
    model = make_model(model_name).fit(X, y)
    if model_name == "rf":
        score = model.predict_proba(x_new[None])[0, 1] - 0.5
    else:
        score = model.decision_function(x_new[None])[0]
    print(f"\nNEW {what} FILE: {os.path.basename(path)}")
    print(f"  Schizophrenia: {'YES' if score > 0 else 'NO'}    (score {score:+.2f}, above 0 = YES)")
    print("  Research demo only. This is not a medical diagnosis.")


# ======================================================================
# 5. MAIN
# ======================================================================
def make_fake_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(2500) / 250
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    eeg = [(rng.standard_normal((19, 2500)) +
            rng.normal(2, .7) * (0.7 if s else 1) * np.sin(2 * np.pi * 10 * t)) for s in y]
    mri = [(rng.uniform(.4, .9) * (0.8 if s else 1) * rng.standard_normal((140, 1))
            + rng.standard_normal((140, 30))) for s in y]
    return np.array([eeg_features(e, 250) for e in eeg]), np.array([mri_features(m) for m in mri]), y


def build(folder, load_and_featurize, what):
    files = find_labelled_files(folder)
    if not files:
        print(f"No labelled {what} files found in {folder}")
        return None, None, None
    X, y, names = [], [], []
    for path, label in files:
        try:
            X.append(load_and_featurize(path))
            y.append(label)
            names.append(stem(path))
        except Exception as e:
            print(f"  could not read {os.path.basename(path)}: {e}")
    sizes = {len(x) for x in X}
    if len(sizes) > 1:
        raise SystemExit(f"{what} files give different feature counts {sorted(sizes)}: the recordings "
                         "have different numbers of channels/ROIs. Use --eeg_channels N, or check the files.")
    return np.array(X), np.array(y), names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", help="folder with eeg/ and/or mri/ (omit for fake demo)")
    ap.add_argument("--model", default="svm", choices=["svm", "lr", "rf"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--fs", type=float, default=250, help="EEG sampling rate for non-EDF files")
    ap.add_argument("--eeg_channels", type=int, help="keep only the first N EEG channels")
    ap.add_argument("--roi_first", action="store_true", help="MRI files are ROIs x time")
    ap.add_argument("--predict_eeg", help="a NEW EEG file: train on all data, then say YES/NO")
    ap.add_argument("--predict_mri", help="a NEW MRI file: train on all data, then say YES/NO")
    a = ap.parse_args()

    if a.data_dir is None:
        print("No --data_dir given: using FAKE data (only tests that the code runs).")
        Xe, Xm, y = make_fake_data()
        names = [f"fake{i:03d}" for i in range(len(y))]
        evaluate(Xe, y, names, "EEG model (fake data)", a.model, a.repeats, "predictions_eeg.csv")
        evaluate(Xm, y, names, "MRI model (fake data)", a.model, a.repeats, "predictions_mri.csv")
    else:
        for what, sub, new_file in (("EEG", "eeg", a.predict_eeg), ("MRI", "mri", a.predict_mri)):
            folder = os.path.join(a.data_dir, sub)
            if not os.path.isdir(folder):
                print(f"\n{what}: folder {folder} not found, skipping.")
                continue
            print(f"\nLoading {what} from {folder} ...")
            if what == "EEG":
                feat = lambda p: eeg_features(*load_eeg(p, a.fs, a.eeg_channels))
            else:
                feat = lambda p: mri_features(load_mri(p, a.roi_first))
            X, y, names = build(folder, feat, what)
            if X is None or len(set(y)) < 2:
                continue
            evaluate(X, y, names, f"{what} model", a.model, a.repeats, f"predictions_{sub}.csv")
            if new_file:
                if stem(new_file) in names:
                    print(f"  note: {stem(new_file)} is also in the training data, so this answer is not a fair test.")
                predict_new(X, y, feat(new_file), a.model, what, new_file)
