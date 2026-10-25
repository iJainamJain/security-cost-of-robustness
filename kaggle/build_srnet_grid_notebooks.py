"""Generate the six SRNet curriculum notebooks (M4_srnet_A ... M4_srnet_F).

    python handoff_M4/code/build_srnet_grid_notebooks.py              # all sessions
    python handoff_M4/code/build_srnet_grid_notebooks.py 256_B12 256_W2  # only these

Each notebook is one Kaggle session: a list of (chain, seed) tasks per GPU,
sized to finish inside the 12-hour cap at the measured 3.9 min/epoch. Together
they cover five methods x three seeds:

    J  J-UNIWARD 0.3 -> 0.2 -> 0.1 -> 0.05      60/25/25/25 epochs   ~8.8 h
    G  GMAS      0.1 -> 0.05                    50/20                ~4.6 h
    R  ROAST     0.1 -> 0.05                    50/20                ~4.6 h
    M  MINICER   0.1 -> 0.05                    50/20                ~4.6 h
    S  SCRIPT (R=0) 0.2 -> 0.1 -> 0.05          60/25/25             ~7.2 h

J, G, R and M read the 5,000-image grid dataset (stego_5000.bin); S reads the
strength-0 SCRIPT dataset (stego_script_r0_5000.bin). Cell code below
deliberately contains no backslash escapes.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent.parent / "results"
EMBED = [("m4.py", HERE / "kaggle_notebook.py"), ("m4_chain.py", HERE / "m4_chain.py"),
         ("m4_srnet.py", HERE / "m4_srnet.py"),
         ("m4_srnet_chain.py", HERE / "m4_srnet_chain.py")]

CHAINS = {
    "J": (["J-UNIWARD_0.3", "J-UNIWARD_0.2", "J-UNIWARD_0.1", "J-UNIWARD_0.05"], [60, 25, 25, 25]),
    "G": (["GMAS_0.1", "GMAS_0.05"], [50, 20]),
    "R": (["ROAST_0.1", "ROAST_0.05"], [50, 20]),
    "M": (["MINICER_0.1", "MINICER_0.05"], [50, 20]),
    "S": (["SCRIPT-R0_0.2", "SCRIPT-R0_0.1", "SCRIPT-R0_0.05"], [60, 25, 25]),
}
# session -> one task list per GPU; a task is (chain key, seed)
SESSIONS = {
    "A": [[("J", 0)], [("J", 1)]],
    "B": [[("J", 2)], [("G", 0), ("R", 0)]],
    "C": [[("M", 0), ("G", 1)], [("R", 1), ("M", 1)]],
    "D": [[("G", 2), ("R", 2)], [("M", 2)]],
    "E": [[("S", 0)], [("S", 1)]],
    "F": [[("S", 2)], []],
    # D and F re-cut on 2026-10-06 to fit what quota each account had left:
    "D1": [[("G", 2)], [("M", 2)]],          # about 4.5 h
    "D2": [[("R", 2)], []],                  # about 4.6 h, grid dataset only
    "D2F": [[("R", 2)], [("S", 2)]],         # about 6.3 h, needs BOTH datasets
}

# 256x256 second-source sessions (exp24 / exp28). Same chains, except that the
# 256x256 sets hold SCRIPT at R=0 for 0.1 and 0.05 only. One source per session:
# attach ONLY that source's dataset, because both sources use the same cell names.
CHAINS["S2"] = (["SCRIPT-R0_0.1", "SCRIPT-R0_0.05"], [50, 20])
SESSIONS_256 = {
    # name: (per-GPU task lists, session limit in hours, expected min/epoch, source)
    # A source makes the notebook refuse to start unless exactly that dataset is
    # attached. The speed test predates the check and is kept as it was run.
    "256_speed": ([[("J", 0), ("S2", 0)], [("G", 0), ("R", 0), ("M", 0)]], 4.6, 1.0, None),
    # M4_srnet_256_speed measured 0.8-1.1 min per epoch (mean 1.0), so one seed
    # of all five chains is about 3.6 h on two T4s and two seeds about 7.3 h.
    "256_B12": ([[("J", 1), ("J", 2), ("S2", 1), ("S2", 2)],
                 [("G", 1), ("R", 1), ("M", 1), ("G", 2), ("R", 2), ("M", 2)]],
                11.2, 1.05, "BOSSbase256"),
    "256_W01": ([[("J", 0), ("J", 1), ("S2", 0), ("S2", 1)],
                 [("G", 0), ("R", 0), ("M", 0), ("G", 1), ("R", 1), ("M", 1)]],
                11.2, 1.05, "BOWS2_256"),
    "256_W2": ([[("J", 2), ("S2", 2)], [("G", 2), ("R", 2), ("M", 2)]], 4.0, 1.05, "BOWS2_256"),
}

# Side-informed reference for SCRIPT (exp36 / exp38): SCRIPT's embedder without
# its coefficient selection, 512x512, same chain shape as SCRIPT at R = 0.
CHAINS["SI"] = (["SI-UERD_0.2", "SI-UERD_0.1", "SI-UERD_0.05"], [60, 25, 25])
SESSIONS_SI = {
    "SI01": [[("SI", 0)], [("SI", 1)]],      # about 7.2 h
    "SI2": [[("SI", 2)], []],                # about 7.2 h, one GPU idle
}
# SCRIPT at R = 0 with the cost repair chosen by exp40 (dataset written by exp40).
CHAINS["SF"] = (["SCRIPT-fix_0.2", "SCRIPT-fix_0.1", "SCRIPT-fix_0.05"], [60, 25, 25])
SESSIONS_SF = {
    "SF01": [[("SF", 0)], [("SF", 1)]],
    "SF2": [[("SF", 2)], []],
}
# The same repair with a longer curriculum (exp42): SF01 / SF2 started at 0.2 and
# SRNet never left chance, which a failed start also looks like. This chain
# starts at 0.6 bpnzac. 165 epochs, about 9.6 h per seed on one T4.
CHAINS["SFH"] = (["SCRIPT-fix_0.6", "SCRIPT-fix_0.4", "SCRIPT-fix_0.3", "SCRIPT-fix_0.2",
                  "SCRIPT-fix_0.1", "SCRIPT-fix_0.05"], [50, 20, 20, 25, 25, 25])
SESSIONS_SFH = {
    "SFH01": [[("SFH", 0)], [("SFH", 1)]],
    "SFH2": [[("SFH", 2)], []],
}

# The like-for-like reference of the repaired SCRIPT (exp43 / exp54): the same
# embedder and floor without the coefficient selection, on the chain of SFH. The
# first stage of SFH was cut by the session-time guard to 21 / 25 / 15 epochs for
# seeds 0 / 1 / 2; each seed here is asked for exactly what its SFH counterpart
# ran, so that the two sides of Table 9 are trained on the same schedule.
_RFH_CHAIN = ["SI-fix_0.6", "SI-fix_0.4", "SI-fix_0.3", "SI-fix_0.2", "SI-fix_0.1", "SI-fix_0.05"]
for _seed, _first in ((0, 21), (1, 25), (2, 15)):
    CHAINS["RFH%d" % _seed] = (_RFH_CHAIN, [_first, 20, 20, 25, 25, 25])
SESSIONS_RFH = {
    "RFH01": [[("RFH0", 0)], [("RFH1", 1)]],      # about 8.7 h
    "RFH2": [[("RFH2", 2)], []],                  # about 8.1 h, one GPU idle
}

# Reference numbers on the same 5,000 images, read from our own result files.
REF = {"EfficientNet-B0": {}, "DCTR+FLD": {}}
for f in (RES / "m4_5k_a/scale_results/cnn/results.json",
          RES / "m4_5k_b/scale_results/cnn/results.json",
          RES / "m4_script_r0_cnn/scale_results/cnn/results.json"):
    if f.exists():
        for r in json.loads(f.read_text()):
            if r.get("status") == "ok":
                REF["EfficientNet-B0"][r["cell"]] = round(r["p_e_mean"], 4)
for f in (RES / "exp16_dctr_fullscale/tier_5000/results.json",
          RES / "exp21_script_low_fullscale/tier_5000/results.json"):
    if f.exists():
        for r in json.loads(f.read_text()):
            REF["DCTR+FLD"][r["cell"]] = round(r["p_e_mean"], 4)


def ref_256(source: str) -> dict:
    """DCTR on the same 256x256 images (exp25), for the summary table."""
    f = RES / "exp24_second_source" / source / "dctr" / "results.json"
    if source == "SI-UERD":                 # 512x512 reference set, DCTR from exp37
        f = RES / "exp36_si_reference" / "tier_5000" / "results.json"
    dctr = {r["cell"]: round(r["p_e_mean"], 4) for r in json.loads(f.read_text())} \
        if f.exists() else {}
    return {"EfficientNet-B0": {}, "DCTR+FLD": dctr}


def build(session: str, gpus: list, limit_h: float = 11.2, mpe: float = 3.9,
          dataset_note: str | None = None, source: str | None = None) -> None:
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": text.strip("\n").splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})

    plan = [[dict(chain=CHAINS[k][0], epochs=CHAINS[k][1], seed=s) for k, s in g] for g in gpus]
    needs_r0 = any(k == "S" for g in gpus for k, _ in g)
    needs_grid = any(k != "S" for g in gpus for k, _ in g)
    data = dataset_note or " and ".join(([("`stego_5000.bin`")] if needs_grid else [])
                                        + (["`stego_script_r0_5000.bin`"] if needs_r0 else []))
    hours = max((sum(sum(CHAINS[k][1]) for k, _ in g) * mpe / 60 for g in gpus), default=0)
    lines = []
    for gi, g in enumerate(gpus):
        lines.append("- GPU %d: %s" % (gi, "; ".join(
            "%s, seed %d" % (" -> ".join(CHAINS[k][0]), s) for k, s in g) or "idle"))
    md("""
# SRNet curriculum, session @S@

@LINES@

**Before you start:** *Accelerator -> GPU T4 x2*, *Internet -> On*, and
*+ Add Input ->* @DATA@. Then **Save Version -> Save & Run All** and close the
tab. Expected run time: about @H@ hours.

Each chain trains at its highest payload first and carries the weights down.
The log prints one line per epoch. The last cell prints the results and writes
`srnet_@S@_results.zip`.
""".replace("@S@", session).replace("@LINES@", "\n".join(lines))
       .replace("@DATA@", data).replace("@H@", "%.1f" % (hours + 0.3)))

    code('''
# ---- Cell 1: preflight -------------------------------------------------
import json
import os
import pathlib
import subprocess
import sys
import time
import zipfile

import torch

NOTEBOOK_T0 = time.time()
SESSION_LIMIT_MIN = @LIMIT@ * 60      # 11.2 = Kaggle's 12-hour cap; less when quota is short
MIN_PER_EPOCH = "@MPE@"            # expected speed, used only for planning
SESSION = "@S@"
EXPECT = "@SRC@"                   # 256x256 source this session is for; "" = the 512x512 grid
PLAN = @PLAN@
REF = @REF@

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
if N_GPU == 0:
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
for i in range(N_GPU):
    print("  GPU", i, torch.cuda.get_device_name(i))
if N_GPU < len([p for p in PLAN if p]):
    print("WARNING: fewer GPUs than planned; the extra task lists run one after "
          "another and may not fit in the session.")

# Each dataset is ONE archive file; unpack it onto the session's local disk.
LOCAL = pathlib.Path("/kaggle/temp" if pathlib.Path("/kaggle/temp").is_dir() else "/tmp") / "stego_data"
print("looking for data under /kaggle/input ...", flush=True)
packed = [p for ext in ("*.bin", "*.zip")
          for p in sorted(pathlib.Path("/kaggle/input").rglob(ext))
          if zipfile.is_zipfile(p)]
if EXPECT and (len(packed) != 1 or EXPECT not in packed[0].name):
    raise SystemExit("This session needs exactly ONE dataset attached: the one holding stego_"
                     + EXPECT + "_5000.bin. Found: "
                     + (", ".join(z.name for z in packed) or "nothing")
                     + ". Fix it under '+ Add Input' and run again.")
for z in packed:
    t_z = time.time()
    dest = LOCAL / z.stem
    print("unpacking", z, "(%.2f GB) ->" % (z.stat().st_size / 1e9), dest, flush=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    print("  done in %.0f s" % (time.time() - t_z), flush=True)

# Find the dataset that holds each chain.
ROOT_OF = {}
roots = [pathlib.Path("/kaggle/input")] + ([LOCAL] if LOCAL.is_dir() else [])
for root in roots:
    for m in sorted(root.rglob("MANIFEST.json")):
        for c in json.loads(m.read_text())["cells"]:
            ROOT_OF.setdefault(c["cell"], str(m.parent))
missing = sorted({c for g in PLAN for t in g for c in t["chain"] if c not in ROOT_OF})
if missing:
    raise SystemExit("not found in any attached dataset: " + ", ".join(missing)
                     + " - check '+ Add Input'.")
for g in PLAN:
    for t in g:
        where = {ROOT_OF[c] for c in t["chain"]}
        if len(where) != 1:
            raise SystemExit("chain spans two datasets: " + ", ".join(t["chain"]))
        t["data"] = where.pop()
        print("  " + " -> ".join(t["chain"]), "| seed", t["seed"], "|", t["data"])

OUT = pathlib.Path("/kaggle/working/srnet_results")
OUT.mkdir(parents=True, exist_ok=True)
'''.replace("@S@", session).replace("@PLAN@", repr(plan))
       .replace("@REF@", repr(ref_256(source) if source else REF))
       .replace("@SRC@", source or "")
       .replace("@LIMIT@", repr(limit_h)).replace("@MPE@", repr(mpe)))

    for name, path in EMBED:
        code("%%writefile " + name + "\n" + path.read_text(encoding="utf-8"))

    code('''
# ---- Cell 2: run the plan, one process per GPU -------------------------
# The cover-vs-cover null test for this loader and split passed in
# M4_srnet_speed (P_E 0.4852, chance 0.4875); it is not repeated here.
budget = SESSION_LIMIT_MIN - (time.time() - NOTEBOOK_T0) / 60 - 15
print("time budget: %.0f min" % budget)
procs = []
for g, tasks in enumerate(PLAN):
    if not tasks:
        continue
    (OUT / ("plan_gpu%d.json" % g)).write_text(json.dumps(tasks))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % N_GPU))
    log = open(OUT / ("srnet_gpu%d.log" % g), "w")
    cmd = [sys.executable, "-u", "m4_srnet_chain.py", "--plan", str(OUT / ("plan_gpu%d.json" % g)),
           "--out", str(OUT / ("part%d" % g)), "--time-budget-min", "%.0f" % budget,
           "--min-per-epoch", MIN_PER_EPOCH]
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
# ---- Cell 3: merge, summarise, package ---------------------------------
import shutil

rows, merged = [], {}
(OUT / "scores").mkdir(parents=True, exist_ok=True)
for g in range(len(PLAN)):
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
(OUT / "seed_results.json").write_text(json.dumps(rows, indent=2))

print("cell".ljust(16) + "seed".rjust(5) + "SRNet".rjust(9) + "transfer".rjust(10)
      + "EffNet-B0".rjust(11) + "DCTR".rjust(8) + "  best epoch (of run)   min/epoch")
print("-" * 86)
for r in sorted(rows, key=lambda r: (r["chain"], r["seed"], r["stage_index"])):
    c = r["cell"]
    e, d = REF["EfficientNet-B0"].get(c), REF["DCTR+FLD"].get(c)
    tr = r["transfer"]["test_p_e"] if r["transfer"] else None
    print(c.ljust(16) + str(r["seed"]).rjust(5) + ("%.4f" % r["test_p_e"]).rjust(9)
          + (("%.4f" % tr) if tr is not None else "-").rjust(10)
          + (("%.4f" % e) if e is not None else "-").rjust(11)
          + (("%.4f" % d) if d is not None else "-").rjust(8)
          + ("%d (%d)" % (r["best_epoch"], r["epochs_run"])).rjust(14)
          + ("%.2f" % r["min_per_epoch"]).rjust(14))
planned = sum(len(t["chain"]) for g in PLAN for t in g)
print()
print("stages finished: %d of %d planned" % (len(rows), planned))
print("transfer = the network from the previous stage applied unchanged")
print("best epoch -1 = fine-tuning never beat the arriving weights on validation")

notes = [
    "# SRNet curriculum, session " + SESSION, "",
    "- Image source: " + (EXPECT or "see the data path in plan_gpu*.json"),
    "- Hardware: %d x %s; torch %s" % (N_GPU, torch.cuda.get_device_name(0), torch.__version__),
    "- SRNet from scratch at the first stage of each chain, then fine-tuned down the "
    "payloads; native-size 8-bit decoded pixels scaled to [0,1], pair constraint, 8 pairs per "
    "batch, Adamax 1e-3 then 1e-4 (after 80% of the epochs in the first stage, 60% in "
    "later stages), L2 2e-4 on kernels, AMP",
    "- One paired 40/10/50 split per seed, shared by all stages of a chain; epoch chosen "
    "on validation, arriving weights included as a candidate",
    "- Null test not repeated: passed in M4_srnet_speed (0.4852, chance 0.4875)",
    "- Stages finished: %d of %d" % (len(rows), planned),
    "- Total minutes: %.0f" % ((time.time() - NOTEBOOK_T0) / 60),
]
(OUT / "NOTES.md").write_text(chr(10).join(notes) + chr(10))

stage = pathlib.Path("/kaggle/working/_zip")
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir()
shutil.copytree(OUT / "scores", stage / "scores")
for f in list(OUT.glob("*.log")) + list(OUT.glob("*.json")) + [OUT / "NOTES.md"]:
    shutil.copy2(f, stage / f.name)
shutil.make_archive("/kaggle/working/srnet_" + SESSION + "_results", "zip", stage)
shutil.rmtree(stage)
print()
print("wrote /kaggle/working/srnet_" + SESSION + "_results.zip - download it from the Output tab")
''')

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / ("M4_srnet_%s.ipynb" % session)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote %s  (%.0f KB, about %.1f h): %s" % (out.name, out.stat().st_size / 1024,
                                                     hours + 0.3, " | ".join(lines)))


ONLY = set(sys.argv[1:])
for s, g in SESSIONS_SI.items():
    if not ONLY or s in ONLY:
        build(s, g, 11.2, 3.9, "ONLY the dataset holding `stego_SI-UERD_5000.bin` - no other dataset",
              "SI-UERD")
for s, g in SESSIONS_SF.items():
    if not ONLY or s in ONLY:
        build(s, g, 11.2, 3.9, "ONLY the dataset holding `stego_SCRIPT-fix_5000.bin` - no other dataset",
              "SCRIPT-fix")
for s, g in SESSIONS_SFH.items():
    if not ONLY or s in ONLY:
        build(s, g, 11.2, 3.5,
              "ONLY the dataset holding `stego_SCRIPT-fix-curriculum_5000.bin` - no other dataset",
              "SCRIPT-fix-curriculum")
for s, g in SESSIONS_RFH.items():
    if not ONLY or s in ONLY:
        build(s, g, 11.2, 3.6,
              "ONLY the dataset holding `stego_SI-fix-curriculum_5000.bin` - no other dataset",
              "SI-fix-curriculum")
for s, g in SESSIONS.items():
    if not ONLY or s in ONLY:
        build(s, g)
for s, (g, limit_h, mpe, source) in SESSIONS_256.items():
    if ONLY and s not in ONLY:
        continue
    note = ("ONLY the dataset holding `stego_%s_5000.bin` - no other dataset" % source
            if source else
            "ONE of the 256x256 datasets (`stego_BOSSbase256_5000.bin` or "
            "`stego_BOWS2_256_5000.bin`) - never both in the same session")
    build(s, g, limit_h, mpe, note, source)
