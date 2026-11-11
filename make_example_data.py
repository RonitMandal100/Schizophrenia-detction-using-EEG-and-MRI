"""Creates data_example/ in the same layout as real data, so you can test the loaders:

    python make_example_data.py
    python sz_real_data.py --data_dir data_example

eeg/ : 14 'h' + 14 'sz' files in EDF format (19 channels, 250 Hz), like the Warsaw set
mri/ : ROI time series as .csv with a header row, plus labels.csv with text labels
The signals are fake. Only the file formats are realistic.
"""
import os
import numpy as np

CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
      "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]


def write_edf(path, data, fs=250, labels=CH):
    """Write channels x samples (microvolts) as a 16-bit EDF file with 1 s records."""
    ns, n_samp = data.shape
    n_rec = n_samp // int(fs)
    pmin, pmax, dmin, dmax = -500.0, 500.0, -32768, 32767
    dig = np.round((data[:, : n_rec * int(fs)] - pmin) / (pmax - pmin) * (dmax - dmin) + dmin)
    dig = np.clip(dig, dmin, dmax).astype("<i2")
    pad = lambda s, w: str(s).ljust(w)[:w]
    head = (pad("0", 8) + pad("X", 80) + pad("X", 80) + pad("01.01.17", 8) + pad("00.00.00", 8)
            + pad(256 * (ns + 1), 8) + pad("", 44) + pad(n_rec, 8) + pad(1, 8) + pad(ns, 4))
    sig = "".join(pad("EEG " + l, 16) for l in labels) + pad("", 80) * ns + pad("uV", 8) * ns
    sig += pad(pmin, 8) * ns + pad(pmax, 8) * ns + pad(dmin, 8) * ns + pad(dmax, 8) * ns
    sig += pad("", 80) * ns + pad(int(fs), 8) * ns + pad("", 32) * ns
    body = np.stack([dig[:, r * int(fs):(r + 1) * int(fs)] for r in range(n_rec)])  # rec x ch x samp
    with open(path, "wb") as f:
        f.write(head.encode("ascii") + sig.encode("ascii") + body.astype("<i2").tobytes())


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    out = "data_example"
    os.makedirs(f"{out}/eeg", exist_ok=True)
    os.makedirs(f"{out}/mri", exist_ok=True)
    t = np.arange(250 * 60) / 250
    for i in range(1, 15):
        for prefix, sz in (("h", 0), ("s", 1)):
            amp = rng.normal(8, 2.5) * (0.7 if sz else 1.0)
            x = 10 * rng.standard_normal((19, t.size)) + amp * np.sin(2 * np.pi * 10 * t)
            write_edf(f"{out}/eeg/{prefix}{i:02d}.edf", x)
    rows = ["subject_id,label"]
    for i in range(40):
        sz = int(i >= 20)
        shared = rng.standard_normal((140, 1)) * rng.uniform(.4, .9) * (0.8 if sz else 1.0)
        ts = shared + rng.standard_normal((140, 30))
        name = f"sub{i:03d}"
        np.savetxt(f"{out}/mri/{name}.csv", ts, delimiter=",", header=",".join(f"roi{j}" for j in range(30)), comments="")
        rows.append(f"{name},{'Patient' if sz else 'Control'}")
    open(f"{out}/mri/labels.csv", "w").write("\n".join(rows) + "\n")
    print(f"Example data written to {out}/")
