"""Generate an SRNet notebook for Kaggle.

    python handoff_M4/code/build_srnet_notebook.py                       # speed test
    python handoff_M4/code/build_srnet_notebook.py --name M4_srnet_A --cells ... --seeds 0 1 2

The default is the SPEED TEST: one seed on two cells, one per GPU, to measure
minutes per epoch and see whether SRNet converges in 40 epochs before any quota
is committed to the full grid. Cell code below deliberately contains no
backslash escapes: they would be interpreted here, not in the notebook.
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="M4_srnet_speed")
ap.add_argument("--cells", nargs="+", default=["GMAS_0.05", "J-UNIWARD_0.2"])
ap.add_argument("--seeds", nargs="+", default=["0"])
ap.add_argument("--epochs", default="40")
ap.add_argument("--pairs", default="8")
ARGS = ap.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent.parent / "results"
OUT = HERE / (ARGS.name + ".ipynb")
EMBED = [("m4.py", HERE / "kaggle_notebook.py"), ("m4_chain.py", HERE / "m4_chain.py"),
         ("m4_srnet.py", HERE / "m4_srnet.py")]

# Reference numbers on the same 5,000 images, read from our own result files.
REF = {"EfficientNet-B0": {}, "DCTR+FLD": {}}
for f in (RES / "m4_5k_a/scale_results/cnn/results.json",
          RES / "m4_5k_b/scale_results/cnn/results.json"):
    if f.exists():
        for r in json.loads(f.read_text()):
            if r.get("status") == "ok":
                REF["EfficientNet-B0"][r["cell"]] = round(r["p_e_mean"], 4)
f = RES / "exp16_dctr_fullscale/tier_5000/results.json"
if f.exists():
    for r in json.loads(f.read_text()):
        REF["DCTR+FLD"][r["cell"]] = round(r["p_e_mean"], 4)
for c in ARGS.cells:
    print("reference", c, {d: REF[d].get(c) for d in REF})

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# SRNet warden: @CELLS@

SRNet is the reference deep steganalyser, trained from scratch with the pair
constraint. This notebook trains it on the cells above with the same split,
epoch selection and P_E as the EfficientNet runs, and prints the two side by
side.

**Before you start:** *Accelerator -> GPU T4 x2*, *Internet -> On*, and
*+ Add Input ->* the 5,000-image dataset (`stego_5000.bin`). Then **Save
Version -> Save & Run All** and close the tab.

The log prints one line per epoch with its time, so the speed is known within
the first half hour.
""".replace("@CELLS@", ", ".join(ARGS.cells)))

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
CELLS = @CELLS@
SEEDS = @SEEDS@
EPOCHS, PAIRS = "@EPOCHS@", "@PAIRS@"
REF = @REF@

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
if N_GPU == 0:
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
for i in range(N_GPU):
    print("  GPU", i, torch.cuda.get_device_name(i))

# The dataset is ONE archive file; unpack it onto the session's local disk.
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

DATA = None
roots = [pathlib.Path("/kaggle/input")] + ([LOCAL] if LOCAL.is_dir() else [])
for root in roots:
    for m in sorted(root.rglob("MANIFEST.json")):
        have = {c["cell"] for c in json.loads(m.read_text())["cells"]}
        if set(CELLS) <= have:
            DATA = m.parent
if DATA is None:
    raise SystemExit("No attached dataset holds all of: " + ", ".join(CELLS)
                     + " - add the 5,000-image dataset via '+ Add Input'.")
mf = json.loads((DATA / "MANIFEST.json").read_text())
print("data root:", DATA, "|", mf["cover_count"], "covers")

OUT = pathlib.Path("/kaggle/working/srnet_results")
OUT.mkdir(parents=True, exist_ok=True)
'''.replace("@CELLS@", repr(ARGS.cells)).replace("@SEEDS@", repr(ARGS.seeds))
   .replace("@EPOCHS@", ARGS.epochs).replace("@PAIRS@", ARGS.pairs)
   .replace("@REF@", repr(REF)))

for name, path in EMBED:
    code("%%writefile " + name + "\n" + path.read_text(encoding="utf-8"))

code('''
# ---- Cell 2: SRNet null test (gate) ------------------------------------
# Covers against other covers, through SRNet's own pair loader. It must read
# chance; if it does not, the grid does not run.
import os

p = subprocess.run([sys.executable, "-u", "m4_srnet.py", "--data", str(DATA),
                    "--epochs", EPOCHS, "--pairs", PAIRS,
                    "--out", str(OUT / "null"), "--null-test"],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
if p.returncode != 0:
    raise RuntimeError("SRNet null test failed - do not run the grid; read the output above.")
print("elapsed: %.0f min" % ((time.time() - NOTEBOOK_T0) / 60))
''')

code('''
# ---- Cell 3: the grid, one process per GPU -----------------------------
parts = [CELLS[i::N_GPU] for i in range(N_GPU)]
budget = SESSION_LIMIT_MIN - (time.time() - NOTEBOOK_T0) / 60 - 15
print("time budget for the grid: %.0f min" % budget)

procs = []
for g, part in enumerate(parts):
    if not part:
        continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    log = open(OUT / ("srnet_gpu%d.log" % g), "w")
    cmd = [sys.executable, "-u", "m4_srnet.py", "--data", str(DATA), "--cells", *part,
           "--seeds", *SEEDS, "--epochs", EPOCHS, "--pairs", PAIRS,
           "--out", str(OUT / ("part%d" % g)), "--time-budget-min", "%.0f" % budget]
    print("GPU", g, "->", ", ".join(part), "| seeds", ", ".join(SEEDS))
    procs.append((g, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

seen = {g: 0 for g, _, _ in procs}
while any(pr.poll() is None for _, pr, _ in procs):
    time.sleep(60)
    for g, _, _ in procs:
        lines = (OUT / ("srnet_gpu%d.log" % g)).read_text(errors="replace").splitlines()
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
# ---- Cell 4: merge, compare, package -----------------------------------
import shutil

import numpy as np

rows, merged = [], {}
(OUT / "scores").mkdir(parents=True, exist_ok=True)
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

summary = []
print("cell".ljust(16) + "SRNet".rjust(8) + "+/-".rjust(8) + "EffNet-B0".rjust(11)
      + "DCTR".rjust(8) + "  min/epoch  best epoch (of run)")
print("-" * 84)
for c in CELLS:
    rr = sorted((r for r in rows if r["cell"] == c), key=lambda r: r["seed"])
    if not rr:
        print(c.ljust(16) + "not run")
        continue
    pe = [r["test_p_e"] for r in rr]
    s = dict(cell=c, method=rr[0]["method"], payload_bpnzac=rr[0]["payload_bpnzac"],
             detector="SRNet", status="ok", seeds=[r["seed"] for r in rr],
             p_e_mean=float(np.mean(pe)), p_e_std=float(np.std(pe)), p_e_per_seed=pe,
             p_e_reversed_mean=float(np.mean([r["test_p_e_reversed"] for r in rr])),
             md_at_fa1_mean=float(np.mean([r["md_at_fa1"] for r in rr])),
             val_p_e_per_seed=[r["val_p_e"] for r in rr],
             best_epoch_per_seed=[r["best_epoch"] for r in rr],
             epochs_run_per_seed=[r["epochs_run"] for r in rr],
             min_per_epoch=float(np.mean([r["min_per_epoch"] for r in rr])),
             p_e_chance_level=rr[0]["p_e_chance_level"], epoch_selection="validation",
             n_train_pairs=rr[0]["n_train_pairs"], n_val_pairs=rr[0]["n_val_pairs"],
             n_test_pairs=rr[0]["n_test_pairs"],
             efficientnet_p_e=REF["EfficientNet-B0"].get(c), dctr_p_e=REF["DCTR+FLD"].get(c))
    summary.append(s)
    e, d = s["efficientnet_p_e"], s["dctr_p_e"]
    print(c.ljust(16) + ("%.4f" % s["p_e_mean"]).rjust(8) + ("%.4f" % s["p_e_std"]).rjust(8)
          + (("%.4f" % e) if e is not None else "-").rjust(11)
          + (("%.4f" % d) if d is not None else "-").rjust(8)
          + ("%.2f" % s["min_per_epoch"]).rjust(11) + "  "
          + ", ".join("%d (%d)" % (r["best_epoch"], r["epochs_run"]) for r in rr))
(OUT / "results.json").write_text(json.dumps(summary, indent=2))
(OUT / "seed_results.json").write_text(json.dumps(rows, indent=2))

null = json.loads((OUT / "null" / "null_test.json").read_text())
print()
print("SRNet null test: P_E %.4f (chance %.4f, floor %.4f)"
      % (null["test_p_e"], null["null_expected"], null["null_floor"]))
if summary:
    mpe = float(np.mean([s["min_per_epoch"] for s in summary]))
    hours = 12 * 3 * int(EPOCHS) * mpe / 60 / max(N_GPU, 1)
    print("speed: %.2f min/epoch -> the full grid (12 cells x 3 seeds x %s epochs) "
          "needs about %.0f session-hours on %d GPUs" % (mpe, EPOCHS, hours, N_GPU))
    print("a best epoch close to the last one means SRNet had not converged")

notes = [
    "# SRNet run notes", "",
    "- Cells: " + ", ".join(CELLS) + "; seeds " + ", ".join(SEEDS),
    "- Hardware: %d x %s; torch %s" % (N_GPU, torch.cuda.get_device_name(0), torch.__version__),
    "- SRNet from scratch, 512x512 8-bit decoded pixels scaled to [0,1], pair constraint, "
    "%s pairs per batch, Adamax 1e-3 then 1e-4 for the last fifth, L2 2e-4 on kernels, AMP, "
    "%s epochs" % (PAIRS, EPOCHS),
    "- One paired 40/10/50 split per seed; epoch chosen on validation",
    "- Null P_E: %.4f (chance %.4f)" % (null["test_p_e"], null["null_expected"]),
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
shutil.make_archive("/kaggle/working/srnet_results", "zip", stage)
shutil.rmtree(stage)
print()
print("wrote /kaggle/working/srnet_results.zip - download it from the Output tab")
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote %s  (%d cells, %.0f KB)" % (OUT, len(cells), OUT.stat().st_size / 1024))
