"""Generate M4_scale_kaggle.ipynb: DCTR + FLD (Part A) then the CNN (Part B),
on a large stego-eval dataset, in one Kaggle session.

    python handoff_M4/code/build_scale_notebook.py

The four source files are embedded verbatim with %%writefile, so the notebook
needs nothing but the attached dataset. Cell code below deliberately contains
no backslash escapes: they would be interpreted here, not in the notebook.
"""
import json
import pathlib

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="M4_scale_kaggle")
ap.add_argument("--seeds", default="3")
ap.add_argument("--epochs", default="25")
ap.add_argument("--batch", default="16")
ap.add_argument("--patience", default="",
                help="early-stopping patience in epochs; empty = none")
ap.add_argument("--cells", nargs="*", default=[],
                help="CNN cells to run; empty = every cell in the manifest")
ARGS = ap.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
OUT = HERE / (ARGS.name + ".ipynb")

EMBED = [("m4.py", HERE / "kaggle_notebook.py"),
         ("dctr.py", SRC / "dctr.py"),
         ("ensemble.py", SRC / "ensemble.py"),
         ("dctr_kaggle.py", HERE / "dctr_kaggle.py")]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Scale run: DCTR + FLD, then CNN

One session, two parts, run back to back:

- **Part A - DCTR + FLD ensemble** (about 1-2 h). The feature-based detector,
  ported from MATLAB to Python so it can run here.
- **Part B - EfficientNet-B0** (about 5 h on two T4s). Same code as the
  300-image run, with more data.

**Before you start:** *Accelerator -> GPU T4 x2*, *Internet -> On*, and
*+ Add Input ->* the `stego-eval-2k` dataset. Then **Save Version -> Save & Run
All** and close the tab.

Part A failing does not stop Part B, and the CNN stops itself cleanly before
the 12-hour cap so nothing finished is lost. The last cell prints both
detectors side by side and writes `scale_results.zip`.

**Reading P_E:** chance is not 0.50. Each part runs its own cover-vs-cover null
test and reports results against that.
""")

code('''
# ---- Cell 1: preflight -------------------------------------------------
import json
import pathlib
import subprocess
import sys
import time

import torch

NOTEBOOK_T0 = time.time()
SESSION_LIMIT_MIN = 11.2 * 60      # Kaggle kills the session at 12 h

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
if N_GPU == 0:
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
for i in range(N_GPU):
    print("  GPU", i, torch.cuda.get_device_name(i))

for pkg in ("timm", "jpeglib"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    print(pkg, __import__(pkg).__version__)

# A dataset shipped as ONE archive file is unpacked here, onto the session's
# local disk. Tens of thousands of loose files in a Kaggle dataset can stall a
# session before the first cell runs; a single file does not, and training then
# reads from local disk instead of the input mount.
import zipfile

LOCAL = pathlib.Path("/kaggle/temp" if pathlib.Path("/kaggle/temp").is_dir() else "/tmp") / "stego_data"
print("looking for data under /kaggle/input ...", flush=True)
packed = [p for ext in ("*.bin", "*.zip")
          for p in sorted(pathlib.Path("/kaggle/input").rglob(ext))
          if zipfile.is_zipfile(p)]
for z in packed:
    t_z = time.time()
    dest = LOCAL / z.stem
    print("unpacking", z, "(%.2f GB) ->" % (z.stat().st_size / 1e9), dest, flush=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    print("  done in %.0f s" % (time.time() - t_z), flush=True)

# If several stego-eval datasets are attached, take the largest.
found = []
roots = [pathlib.Path("/kaggle/input")] + ([LOCAL] if LOCAL.is_dir() else [])
for root in roots:
    for m in root.rglob("MANIFEST.json"):
        found.append((json.loads(m.read_text())["cover_count"], m.parent))
if not found:
    raise SystemExit("No MANIFEST.json under /kaggle/input - add the dataset "
                     "via '+ Add Input'.")
found.sort(key=lambda t: t[0])
N_COVERS, DATA = found[-1]
print("data root:", DATA, "|", N_COVERS, "covers")
mf = json.loads((DATA / "MANIFEST.json").read_text())
for c in mf["cells"]:
    print("  " + c["cell"].ljust(18) + str(c["count"]).rjust(6) + " images   " + c["channel"])

OUT = pathlib.Path("/kaggle/working/scale_results")
OUT.mkdir(parents=True, exist_ok=True)
''')

for name, path in EMBED:
    code("%%writefile " + name + "\n" + path.read_text(encoding="utf-8"))

code('''
# ---- Cell 2: Part A - DCTR + FLD ---------------------------------------
# Features for covers and all cells on GPU 0, then an FLD ensemble with five
# paired 50/50 splits per cell. Includes its own cover-vs-cover null test.
# A failure here is reported loudly but does not stop Part B.
#
# RUN_PART_A = False skips it. Version 1 of this notebook already produced the
# DCTR results for the 2,000-image dataset (3.3 h), so a rerun for the CNN
# should not pay for them again. Set it to True for a new dataset.
RUN_PART_A = False
PART_A_OK = False
if RUN_PART_A:
    p = subprocess.run([sys.executable, "dctr_kaggle.py", "--data", str(DATA),
                        "--out", str(OUT / "dctr"), "--device", "cuda:0"])
    PART_A_OK = p.returncode == 0
    print()
    print("PART A " + ("finished" if PART_A_OK else
                       "FAILED (exit " + str(p.returncode) + ") - continuing to Part B"))
else:
    print("PART A skipped (RUN_PART_A = False)")
print("elapsed: %.0f min" % ((time.time() - NOTEBOOK_T0) / 60))
''')

code('''
# ---- Cell 3: Part B - CNN null test (gate) -----------------------------
# Covers against other covers. The script prints the chance level for its
# test size and the pass floor, and exits non-zero below the floor - in which
# case this cell raises and the CNN grid does not run.
CNN = OUT / "cnn"
SEEDS, EPOCHS, BATCH, PATIENCE = "@SEEDS@", "@EPOCHS@", "@BATCH@", "@PATIENCE@"
CNN_CELLS = @CELLS@          # empty = every cell in the manifest
COMMON = ["--data", str(DATA), "--seeds", SEEDS, "--epochs", EPOCHS,
          "--batch-size", BATCH]
if PATIENCE:
    COMMON += ["--patience", PATIENCE]
print("CNN settings: seeds", SEEDS, "| max epochs", EPOCHS, "| batch", BATCH,
      "| patience", PATIENCE or "none", "|", len(CNN_CELLS) or "all", "cells")

p = subprocess.run([sys.executable, "m4.py", *COMMON, "--out", str(CNN),
                    "--null-test"])
if p.returncode != 0:
    raise RuntimeError("CNN null test failed - the pipeline is leaking. "
                       "Do not run the grid; read the output above.")
print("elapsed: %.0f min" % ((time.time() - NOTEBOOK_T0) / 60))
'''.replace("@SEEDS@", ARGS.seeds).replace("@EPOCHS@", ARGS.epochs)
   .replace("@BATCH@", ARGS.batch).replace("@PATIENCE@", ARGS.patience)
   .replace("@CELLS@", repr(ARGS.cells)))

code('''
# ---- Cell 4: Part B - CNN grid, one process per GPU --------------------
# Cells are dealt alternately to the GPUs. Each process checkpoints after
# every cell and stops cleanly if the next cell would not fit in the time left.
import os

names = list(CNN_CELLS) or [c["cell"] for c in mf["cells"]]
unknown = set(names) - {c["cell"] for c in mf["cells"]}
if unknown:
    raise SystemExit("cells not in the manifest: " + ", ".join(sorted(unknown)))
parts = [names[i::N_GPU] for i in range(N_GPU)]
budget = SESSION_LIMIT_MIN - (time.time() - NOTEBOOK_T0) / 60 - 15
print("time budget for the CNN grid: %.0f min" % budget)

procs = []
for g, part in enumerate(parts):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    log = open(OUT / ("cnn_gpu%d.log" % g), "w")
    cmd = [sys.executable, "-u", "m4.py", *COMMON, "--out", str(OUT / ("cnn_part%d" % g)),
           "--cells", *part, "--time-budget-min", "%.0f" % budget]
    print("GPU", g, "->", len(part), "cells:", ", ".join(part))
    procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

# Echo progress lines while waiting, so the log shows the run is alive.
seen = [0] * len(procs)
while any(pr.poll() is None for pr, _ in procs):
    time.sleep(60)
    for g in range(len(procs)):
        lines = (OUT / ("cnn_gpu%d.log" % g)).read_text(errors="replace").splitlines()
        for ln in lines[seen[g]:]:
            if "Warning" in ln or "warn" in ln or "autocast" in ln or "GradScaler" in ln:
                continue
            print("[gpu%d] %s" % (g, ln), flush=True)
        seen[g] = len(lines)
for g, (pr, log) in enumerate(procs):
    log.close()
    print("GPU", g, "process exit code", pr.returncode)
print("elapsed: %.0f min" % ((time.time() - NOTEBOOK_T0) / 60))
''')

code('''
# ---- Cell 5: merge, compare, package -----------------------------------
import shutil

# Merge the per-GPU CNN outputs into one results.json and one scores folder.
cnn_res = []
(CNN / "scores").mkdir(parents=True, exist_ok=True)
for g in range(N_GPU):
    part = OUT / ("cnn_part%d" % g)
    if (part / "results.json").exists():
        cnn_res += json.loads((part / "results.json").read_text())
    if (part / "scores").is_dir():
        for f in (part / "scores").glob("*.csv"):
            shutil.copy2(f, CNN / "scores" / f.name)
(CNN / "results.json").write_text(json.dumps(cnn_res, indent=2))

dctr_res = []
if (OUT / "dctr" / "results.json").exists():
    dctr_res = json.loads((OUT / "dctr" / "results.json").read_text())
D = {r["cell"]: r for r in dctr_res if r["status"] == "ok"}
C = {r["cell"]: r for r in cnn_res if r["status"] == "ok"}

def null_of(folder):
    f = OUT / folder / "null_test.json"
    return json.loads(f.read_text()) if f.exists() else None

nd, nc = null_of("dctr"), null_of("cnn")
print("null tests (cover vs cover):")
if nd:
    print("  DCTR+FLD  P_E %.4f at %d test pairs" % (nd["p_e_mean"], nd["n_test_pairs"]))
if nc:
    print("  CNN       P_E %.4f at %d test pairs (chance %.4f, floor %.4f)"
          % (nc["p_e_mean"], nc["n_test_pairs"], nc["null_expected"], nc["null_floor"]))
print()
print("cell".ljust(18) + "DCTR+FLD".rjust(10) + "+/-".rjust(8)
      + "CNN".rjust(10) + "+/-".rjust(8) + "  lower")
print("-" * 62)
for c in mf["cells"]:
    n = c["cell"]
    d, k = D.get(n), C.get(n)
    line = n.ljust(18)
    line += ("%.4f" % d["p_e_mean"]).rjust(10) + ("%.4f" % d["p_e_std"]).rjust(8) if d else "-".rjust(10) + "-".rjust(8)
    line += ("%.4f" % k["p_e_mean"]).rjust(10) + ("%.4f" % k["p_e_std"]).rjust(8) if k else "-".rjust(10) + "-".rjust(8)
    if d and k:
        line += "  " + ("CNN" if k["p_e_mean"] < d["p_e_mean"] - 0.005 else
                        "DCTR" if d["p_e_mean"] < k["p_e_mean"] - 0.005 else "tie")
    print(line)

missing_cnn = [c["cell"] for c in mf["cells"] if c["cell"] not in C]
missing_dctr = [c["cell"] for c in mf["cells"] if c["cell"] not in D]
print()
print("DCTR cells done: %d/%d" % (len(D), len(mf["cells"])))
print("CNN  cells done: %d/%d" % (len(C), len(mf["cells"])))
if missing_cnn:
    print("CNN cells NOT run (time budget or failure):", ", ".join(missing_cnn))
if missing_dctr:
    print("DCTR cells NOT run:", ", ".join(missing_dctr))

# Inverted-score check for the CNN (inverted scores read as ~0.5, 'secure').
for n, r in C.items():
    if (r["p_e_reversed_mean"] < r["p_e_chance_level"] - 0.06
            and r["p_e_reversed_mean"] < r["p_e_mean"] - 0.05):
        print("  CHECK: " + n + " CNN scores look INVERTED")

notes = [
    "# Scale run notes", "",
    "- Dataset: %d covers, %d cells" % (N_COVERS, len(mf["cells"])),
    "- Hardware: %d x %s" % (N_GPU, torch.cuda.get_device_name(0)),
    "- torch %s" % torch.__version__,
    "- Part A: DCTR (Python port, validated against the MATLAB reference) + FLD "
    "ensemble, d_sub 800, L 200, 5 paired 50/50 splits",
    "- Part B: EfficientNet-B0, ImageNet-pretrained, 1 channel, 512x512, no resize; "
    "%s seeds, %s epochs, batch %s, AdamW lr 1e-4, AMP; paired 40/10/50 split, "
    "epoch chosen on validation" % (SEEDS, EPOCHS, BATCH),
    "- DCTR null P_E: %s" % ("%.4f" % nd["p_e_mean"] if nd else "NOT RUN"),
    "- CNN null P_E: %s" % ("%.4f" % nc["p_e_mean"] if nc else "NOT RUN"),
    "- DCTR cells done: %d/%d; CNN cells done: %d/%d"
    % (len(D), len(mf["cells"]), len(C), len(mf["cells"])),
    "- Total minutes: %.0f" % ((time.time() - NOTEBOOK_T0) / 60),
]
(OUT / "NOTES.md").write_text(chr(10).join(notes) + chr(10))

# The zip holds results, scores, logs and notes. The DCTR feature matrices
# (about 1.2 GB) stay as separate output files: download them only if needed.
stage = pathlib.Path("/kaggle/working/_zip")
if stage.exists():
    shutil.rmtree(stage)
for sub in ("dctr", "cnn"):
    if (OUT / sub).is_dir():
        shutil.copytree(OUT / sub, stage / sub,
                        ignore=shutil.ignore_patterns("features"))
for f in OUT.glob("*.log"):
    shutil.copy2(f, stage / f.name)
shutil.copy2(OUT / "NOTES.md", stage / "NOTES.md")
shutil.make_archive("/kaggle/working/scale_results", "zip", stage)
shutil.rmtree(stage)
print()
print("wrote /kaggle/working/scale_results.zip - download it from the Output tab")
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote %s  (%d cells, %.0f KB)" % (OUT, len(cells), OUT.stat().st_size / 1024))
