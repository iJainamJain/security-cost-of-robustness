# The Security Cost of Robustness — code and results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23639889.svg)](https://doi.org/10.5281/zenodo.23639889)

Code, patches and result files for:

> J. Jain, S. P. Patil, A. S. Deshmukh, H. M. Shinde, N. Mahajan,
> *The Security Cost of Robustness: Measuring Robust JPEG Steganography on Two Recompression Channels*,
> manuscript submitted for publication, 2026.

The paper evaluates four robust JPEG steganography schemes (GMAS, ROAST, MINICER, SCRIPT) and
J-UNIWARD on one protocol. It covers 10,000 BOSSbase images with DCTR + FLD, and 5,000 images
with EfficientNet-B0 and SRNet, on the two recompression channels the schemes target. Every
number in the paper's tables comes from a result file in `results/`. Every figure is drawn from
those files by a script, with no numbers typed by hand.

## Layout

```
src/                 shared code: channels, DCTR port, FLD ensemble, P_E and other metrics, paths
experiments/         one script per result (index in experiments/README.md); MATLAB embedding shards
kaggle/              GPU detectors run on Kaggle: EfficientNet-B0 and SRNet notebooks and their builders
figures/             generate_paper_figures.py draws Figures 1, 3 and 5-10; experiments/exp30 draws 2 and 4
baselines/patches/   our changes to the reference implementations, as patches
baselines/patched_readme/  what each patch fixes and why
setup_baselines.py   clones the reference implementations and applies the patches
results/             result files behind every table and figure (JSON, per-image embedding logs)
```

## Reproducing

**Check the published numbers (Python only).** These run on the shipped result files:

```
pip install -r requirements.txt
python experiments/exp29_paper_tables.py      # rebuilds results/exp29_paper_tables/ (paper tables)
python experiments/exp30_paper_figures.py
python figures/generate_paper_figures.py
```

`exp29` reproduces `paper_tables.json` and `paper_tables.md` byte for byte from the shipped
results.

**Re-run the experiments.** This needs MATLAB (we used R2024b), the image sets and the
reference implementations.

1. **Images.** BOSSbase 1.01 is at http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip.
   BOWS2 is available with BOSSbase on Kaggle at `kaggle.com/datasets/zapak1010/bossbase-bows2`.
   Unpack them into `data/BOSSbase_1.01/` and `data/BOWS2/`, then make the JPEG covers with
   `python experiments/prep_covers.py --qf 65`.
2. **Reference implementations.** Run `python setup_baselines.py`. It clones
   https://github.com/coriverchen/Robust_Steganography at commit `f1765807` into
   `baselines/Robust_Steganography/`, then builds `baselines/patched/` from it and our patches.
   The clone is never modified; the MATLAB shards in `experiments/` put the patched files first
   on the path. That repository carries no licence, so it is not redistributed here.
3. **Feature extractors.** DCTR and CC-PEV are from the DDE Laboratory, Binghamton University
   (https://dde.binghamton.edu/download/). `src/dctr.py` is a Python port checked against the
   MATLAB extractor by `experiments/exp15_dctr_port_check.py`.
4. **Embedding and detection.** `exp13_fullscale.py` embeds every cell (sharded MATLAB runs).
   `exp16_dctr_fullscale.py` and `exp21_script_low_fullscale.py` run DCTR + FLD. The networks
   run on Kaggle from the notebooks in `kaggle/`; their result files go under `results/m4_*`.
   `exp29_paper_tables.py` then rebuilds the tables.

## Where each result comes from

| Paper | Scripts |
|---|---|
| Main security and delivery tables, Figures 2 and 4 | `exp13`, `exp16`, `exp21`, `kaggle/`, `exp29`, `exp30` |
| Chance level of the paired test | `exp44_paired_permutation.py` |
| Same-quality-factor channel: SCRIPT strength, MINICER step | `exp21`, `exp50_minicer_step_sweep.py` |
| Three detectors at 5,000 images, tuned ensemble | `exp16`, `exp26_tuned_ensemble.py`, `kaggle/` |
| SCRIPT diagnosis (zero-cost coefficients) | `exp32`–`exp35`, `exp45`, `exp46`, `exp51` |
| SCRIPT repair and the side-informed reference | `exp40`, `exp42`, `exp43`, `exp54`, `exp36`–`exp38` |
| Dataset size and subset choice | `exp49_random_subset.py`, `exp29` |
| Second image source and second quality factor | `exp24`, `exp25`, `exp28`, `exp41`, `exp53` |
| Appendix A: spreading the payload over several images | `exp17`, `exp22`, `exp31` |
| Reproducibility of the reference implementations | `exp02`–`exp06`, `exp48`, `baselines/patched_readme/` |
| Figures 1, 3 and 5–10 | `figures/generate_paper_figures.py`, `exp55_figure_data.py` |

Not included: the stego images and feature files (tens of GB; every script that makes them is
here), the BOSSbase and BOWS2 images, DDE's MATLAB code, and the upstream implementations.

## Licence

Our code is released under the MIT licence (`LICENSE`). The context lines inside
`baselines/patches/` quote the upstream implementations, which remain their authors' work.

## Citation

See `CITATION.cff`. Once the paper is published, please cite the paper as well as this archive.
