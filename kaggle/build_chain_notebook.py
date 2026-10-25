"""Generate M4_chain_cnn.ipynb: curriculum training of the CNN warden.

    python handoff_M4/code/build_chain_notebook.py

The chain's cells may sit in different attached datasets (SCRIPT 0.05 is in the
5,000-image grid dataset, 0.02 and 0.01 in the low-payload one); the notebook
finds each by name. Cell code below deliberately contains no backslash escapes:
they would be interpreted here, not in the notebook.
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="M4_chain_cnn")
ap.add_argument("--chain", nargs="+", default=["SCRIPT_0.05", "SCRIPT_0.02", "SCRIPT_0.01"])
ap.add_argument("--epochs", nargs="+", default=["40", "25", "25"])
ap.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3"])
ap.add_argument("--batch", default="16")
ARGS = ap.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / (ARGS.name + ".ipynb")
EMBED = [("m4.py", HERE / "kaggle_notebook.py"), ("m4_chain.py", HERE / "m4_chain.py")]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# CNN warden with curriculum training: @CHAIN@

Trained from scratch at 0.01 bpnzac the CNN learned nothing (P_E 0.496, chance
0.491). A real warden would train where the signal is learnable and carry the
weights down. This notebook does that, one stage after another, with one paired
split per seed shared by every stage.

**Before you start:** *Accelerator -> GPU T4 x2*, *Internet -> On*, and
*+ Add Input ->* **both** datasets: the 5,000-image grid (`stego_5000.bin`) and
the low-payload one (`stego_script_low_5000.bin`). Then **Save Version -> Save &
Run All** and close the tab. Expected run time: about 4 hours.

The last cell prints the result per stage and writes `chain_results.zip`.
""".replace("@CHAIN@", " -> ".join(ARGS.chain)))

code('''
# ---- Cell 1: preflight -------------------------------------------------
import json
import pathlib
import subprocess
import sys
import time
import zipfile

import torch

NOTEBOOK_T0 = time.time()
SESSION_LIMIT_MIN = 11.2 * 60      # Kaggle kills the session at 12 h
CHAIN = @CHAIN@
EPOCHS = @EPOCHS@
SEEDS = @SEEDS@
BATCH = "@BATCH@"

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
if N_GPU == 0:
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
for i in range(N_GPU):
    print("  GPU", i, torch.cuda.get_device_name(i))
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm
print("timm", timm.__version__)

# Each dataset is ONE archive file; unpack it onto the session's local disk.
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

# Find every cell of the chain, whichever dataset it is in.
WHERE = {}
roots = [pathlib.Path("/kaggle/input")] + ([LOCAL] if LOCAL.is_dir() else [])
for root in roots:
    for m in sorted(root.rglob("MANIFEST.json")):
        mf = json.loads(m.read_text())
        for c in mf["cells"]:
            if c["cell"] in CHAIN and c["cell"] not in WHERE:
                WHERE[c["cell"]] = (m.parent, m.parent / mf["cover_dir"], m.parent / c["stego_dir"])
missing = [c for c in CHAIN if c not in WHERE]
if missing:
    raise SystemExit("not found in any attached dataset: " + ", ".join(missing)
                     + " - attach BOTH datasets via '+ Add Input'.")
for c in CHAIN:
    print("  " + c.ljust(14), WHERE[c][2])

# The datasets must hold the SAME covers, or a pair would not be a pair.
DATA, COVER, _ = WHERE[CHAIN[-1]]
ref = {p.name: p.stat().st_size for p in COVER.glob("*.jpg")}
for c in CHAIN:
    other = {p.name: p.stat().st_size for p in WHERE[c][1].glob("*.jpg")}
    if other != ref:
        raise SystemExit("cover folders differ between datasets (" + c + ") - stop.")
print("covers:", COVER, "|", len(ref), "images, identical across datasets")

OUT = pathlib.Path("/kaggle/working/chain_results")
OUT.mkdir(parents=True, exist_ok=True)
'''.replace("@CHAIN@", repr(ARGS.chain)).replace("@EPOCHS@", repr(ARGS.epochs))
   .replace("@SEEDS@", repr(ARGS.seeds)).replace("@BATCH@", ARGS.batch))

for name, path in EMBED:
    code("%%writefile " + name + "\n" + path.read_text(encoding="utf-8"))

code('''
# ---- Cell 2: null test (gate) ------------------------------------------
# Covers against other covers must read chance. If this fails the pipeline is
# leaking and the chain does not run.
p = subprocess.run([sys.executable, "m4.py", "--data", str(DATA), "--seeds", "3",
                    "--epochs", "40", "--batch-size", BATCH,
                    "--out", str(OUT / "null"), "--null-test"])
if p.returncode != 0:
    raise RuntimeError("CNN null test failed - do not run the chain; read the output above.")
print("elapsed: %.0f min" % ((time.time() - NOTEBOOK_T0) / 60))
''')

code('''
# ---- Cell 3: the chain, one process per GPU ----------------------------
# Seeds are dealt alternately to the GPUs. Each process checkpoints after every
# seed and stops cleanly if the next seed would not fit in the time left.
import os

parts = [SEEDS[i::N_GPU] for i in range(N_GPU)]
budget = SESSION_LIMIT_MIN - (time.time() - NOTEBOOK_T0) / 60 - 15
print("time budget for the chain: %.0f min" % budget)
stage_args = []
for c in CHAIN:
    stage_args += ["--stage", c + "=" + str(WHERE[c][2])]

procs = []
for g, part in enumerate(parts):
    if not part:
        continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    log = open(OUT / ("chain_gpu%d.log" % g), "w")
    cmd = [sys.executable, "-u", "m4_chain.py", "--cover", str(COVER), *stage_args,
           "--epochs", *EPOCHS, "--seeds", *part, "--batch-size", BATCH,
           "--out", str(OUT / ("part%d" % g)), "--time-budget-min", "%.0f" % budget]
    print("GPU", g, "-> seeds", ", ".join(part))
    procs.append((g, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

seen = {g: 0 for g, _, _ in procs}
while any(pr.poll() is None for _, pr, _ in procs):
    time.sleep(60)
    for g, _, _ in procs:
        lines = (OUT / ("chain_gpu%d.log" % g)).read_text(errors="replace").splitlines()
        for ln in lines[seen[g]:]:
            if "Warning" in ln or "warn" in ln or "autocast" in ln or "GradScaler" in ln:
                continue
            print("[gpu%d] %s" % (g, ln), flush=True)
        seen[g] = len(lines)
for g, pr, log in procs:
    log.close()
    print("GPU", g, "process exit code", pr.returncode)
print("elapsed: %.0f min" % ((time.time() - NOTEBOOK_T0) / 60))
''')

code('''
# ---- Cell 4: merge, summarise, package ---------------------------------
import shutil

import numpy as np

rows = []
(OUT / "scores").mkdir(parents=True, exist_ok=True)
merged = {}
for g in range(N_GPU):
    part = OUT / ("part%d" % g)
    if (part / "seed_results.json").exists():
        rows += json.loads((part / "seed_results.json").read_text())
    if (part / "scores").is_dir():
        for f in sorted((part / "scores").glob("*.csv")):
            lines = f.read_text().splitlines()
            merged.setdefault(f.name, [lines[0]])
            merged[f.name] += lines[1:]
for name, lines in merged.items():
    (OUT / "scores" / name).write_text(chr(10).join(lines) + chr(10))

# From-scratch reference on the same 5,000 images (earlier runs).
SCRATCH = {"SCRIPT_0.05": 0.2358, "SCRIPT_0.02": 0.3963, "SCRIPT_0.01": 0.4959}
DCTR = {"SCRIPT_0.05": 0.2304, "SCRIPT_0.02": 0.4031, "SCRIPT_0.01": 0.4723}

summary = []
print("stage".ljust(14) + "chain P_E".rjust(11) + "+/-".rjust(8) + "transfer".rjust(10)
      + "scratch".rjust(9) + "DCTR".rjust(8) + "  caught at 1% FA   best epochs")
print("-" * 92)
for c in CHAIN:
    rr = sorted((r for r in rows if r["stage"] == c), key=lambda r: r["seed"])
    if not rr:
        continue
    pe = [r["test_p_e"] for r in rr]
    tr = [r["transfer"]["test_p_e"] for r in rr if r["transfer"]]
    caught = [1 - r["md_at_fa1"] for r in rr]
    s = dict(stage=c, seeds=[r["seed"] for r in rr], p_e_mean=float(np.mean(pe)),
             p_e_std=float(np.std(pe)), p_e_per_seed=pe,
             transfer_p_e_mean=float(np.mean(tr)) if tr else None,
             caught_at_fa1_mean=float(np.mean(caught)),
             best_epoch_per_seed=[r["best_epoch"] for r in rr],
             val_p_e_per_seed=[r["val_p_e"] for r in rr],
             n_train_pairs=rr[0]["n_train_pairs"], n_val_pairs=rr[0]["n_val_pairs"],
             n_test_pairs=rr[0]["n_test_pairs"], epoch_selection="validation",
             from_scratch_p_e=SCRATCH.get(c), dctr_p_e=DCTR.get(c))
    summary.append(s)
    print(c.ljust(14) + ("%.4f" % s["p_e_mean"]).rjust(11) + ("%.4f" % s["p_e_std"]).rjust(8)
          + (("%.4f" % s["transfer_p_e_mean"]) if tr else "-").rjust(10)
          + (("%.4f" % SCRATCH[c]) if c in SCRATCH else "-").rjust(9)
          + (("%.4f" % DCTR[c]) if c in DCTR else "-").rjust(8)
          + ("%.3f" % s["caught_at_fa1_mean"]).rjust(12) + "        "
          + str(s["best_epoch_per_seed"]))
(OUT / "chain_results.json").write_text(json.dumps(summary, indent=2))
(OUT / "seed_results.json").write_text(json.dumps(rows, indent=2))

done = sorted({r["seed"] for r in rows if r["stage"] == CHAIN[-1]})
print()
print("seeds finished:", done, "of", SEEDS)
print("transfer = the network from the previous stage applied unchanged")
print("best epoch -1 = fine-tuning never beat the arriving weights on validation")

null = json.loads((OUT / "null" / "null_test.json").read_text())
notes = [
    "# Chain run notes", "",
    "- Chain: " + " -> ".join(CHAIN) + ", epochs " + "/".join(EPOCHS),
    "- Hardware: %d x %s; torch %s" % (N_GPU, torch.cuda.get_device_name(0), torch.__version__),
    "- EfficientNet-B0, ImageNet-pretrained, 1 channel, 512x512, no resize; batch %s, "
    "AdamW lr 1e-4 (fresh optimiser per stage), AMP" % BATCH,
    "- One paired 40/10/50 split per seed, shared by all stages; epoch chosen on "
    "validation, arriving weights included as a candidate",
    "- CNN null P_E: %.4f (chance %.4f)" % (null["p_e_mean"], null["null_expected"]),
    "- Seeds finished: %s of %s" % (done, SEEDS),
    "- Total minutes: %.0f" % ((time.time() - NOTEBOOK_T0) / 60),
]
(OUT / "NOTES.md").write_text(chr(10).join(notes) + chr(10))

stage = pathlib.Path("/kaggle/working/_zip")
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir()
shutil.copytree(OUT / "scores", stage / "scores")
shutil.copytree(OUT / "null", stage / "null")
for f in list(OUT.glob("*.log")) + list(OUT.glob("*.json")) + [OUT / "NOTES.md"]:
    shutil.copy2(f, stage / f.name)
shutil.make_archive("/kaggle/working/chain_results", "zip", stage)
shutil.rmtree(stage)
print()
print("wrote /kaggle/working/chain_results.zip - download it from the Output tab")
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote %s  (%d cells, %.0f KB)" % (OUT, len(cells), OUT.stat().st_size / 1024))
