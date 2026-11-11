import os, tempfile, numpy as np
from scipy.io import savemat
from sz_real_data import read_edf, pick_standard_19, read_matrix, find_labelled_files, load_mri, load_eeg, STD_19
from make_example_data import write_edf, CH

rng = np.random.default_rng(0)

def test_edf_round_trip():
    x = rng.normal(0, 50, (19, 250 * 5))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.edf"); write_edf(p, x)
        y, fs, names = read_edf(p)
    assert fs == 250 and y.shape == x.shape and names[0] == "EEG Fp1"
    assert np.abs(y - x).max() < 0.05          # only 16-bit rounding error

def test_channel_reorder():
    labels = CH[::-1]                           # file stores channels in reverse order
    x = np.arange(19)[:, None] * np.ones((19, 500))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.edf"); write_edf(p, x, labels=labels)
        y, fs, names = read_edf(p)
    z = pick_standard_19(y, names)
    assert np.allclose(z[:, 0], 18 - np.arange(19), atol=0.05)   # Fp1 was last in the file -> 18

def test_text_formats_and_header():
    ts = rng.normal(size=(100, 12))
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/a.npy", ts); savemat(f"{d}/a.mat", {"ROI": ts, "note": np.array([[1.0]])})
        np.savetxt(f"{d}/a.csv", ts, delimiter=",", header=",".join(f"r{i}" for i in range(12)), comments="")
        np.savetxt(f"{d}/a.1D", ts, delimiter="\t", header="# afni", comments="")
        np.savetxt(f"{d}/a.txt", ts)
        for ext in ("npy", "mat", "csv", "1D", "txt"):
            assert np.allclose(read_matrix(f"{d}/a.{ext}"), ts, atol=1e-5), ext
        assert load_mri(f"{d}/a.npy", roi_first=False).shape == (100, 12)
        assert load_mri(f"{d}/a.npy", roi_first=True).shape == (12, 100)

def test_samples_by_channels_is_transposed():
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/a.npy", rng.normal(size=(5000, 19)))
        data, fs = load_eeg(f"{d}/a.npy", 250, None)
    assert data.shape == (19, 5000)

def test_label_sources():
    with tempfile.TemporaryDirectory() as d:
        for sub, f in [("sz", "x1.npy"), ("control", "x2.npy"), (".", "s05.npy"), (".", "h05.npy"), (".", "mystery.npy")]:
            os.makedirs(f"{d}/{sub}", exist_ok=True); open(f"{d}/{sub}/{f}", "w").close()
        got = {os.path.basename(p): l for p, l in find_labelled_files(d)}
    assert got == {"x1.npy": 1, "x2.npy": 0, "s05.npy": 1, "h05.npy": 0}, got

def test_labels_csv_with_nii_gz_names():
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/0040001.nii.gz", "w").close(); open(f"{d}/0040002.nii.gz", "w").close()
        open(f"{d}/labels.csv", "w").write("id,group\n0040001,Patient\n0040002,Control\n")
        got = {os.path.basename(p): l for p, l in find_labelled_files(d)}
    assert got == {"0040001.nii.gz": 1, "0040002.nii.gz": 0}, got

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
