# experiments/

One script per result. The first line of each script's docstring:

| Script | Purpose |
|---|---|
| `dctr_sets.py` | Shared plumbing for the later DCTR experiments (exp40, exp41): chunked feature |
| `exp02_m1_gate.py` | M1 gate driver: invoke MATLAB non-interactively and record the outcome. |
| `exp03_grid_runner.py` | Sharded MATLAB runner — parallelism without Parallel Computing Toolbox. |
| `exp04_overflow_analysis.py` | Why GMAS is not errorless on real covers, and why the bundled demo hides it. |
| `exp05_roast_reconstruction.py` | ROAST: shipped driver vs reconstructed pipeline, head to head. |
| `exp06_roast_sweep.py` | Does the ROAST reconstruction hold beyond the single point it was found at? |
| `exp07_pe_gate.py` | M2 gate: does our P_E pipeline produce trustworthy security numbers? |
| `exp08_arm_a.py` | M3 Arm A — the security-vs-robustness Pareto front. |
| `exp10_arm_b.py` | M3 Arm B — the same-QF methods, on the channel they were designed for. |
| `exp11_ccpev_crosscheck.py` | Cross-check the Arm A security results with a second, independent feature set. |
| `exp12_figures.py` | Paper figures, built only from results files (no hand-typed numbers). |
| `exp13_fullscale.py` | Full-scale stego generation (journal roadmap WP1 / Paper 2 groundwork). |
| `exp14_pack_tier.py` | Package one tier of the full-scale run as a Kaggle dataset zip. |
| `exp15_dctr_port_check.py` | Gate for the Python DCTR port: it must reproduce the MATLAB features. |
| `exp16_dctr_fullscale.py` | DCTR + FLD ensemble over the full-scale stego set (exp13), on this laptop. |
| `exp17_batch_pooling.py` | Paper 2, first test: does spreading a payload over several covers help? |
| `exp18_low_payload.py` | Paper 2, decisive test: does robust embedding become undetectable as the |
| `exp19_followups.py` | Three cheap follow-up tests, queued behind exp18 (which waits for exp16). |
| `exp20_mismatched_warden.py` | Stress test for exp18's good news: a warden who does NOT train at the |
| `exp21_script_low_fullscale.py` | SCRIPT at full scale: the low-payload scale check and Paper 1's corrected rows. |
| `exp22_cnn_warden_pooling.py` | Paper 2: is the CNN a stronger batch warden than DCTR on spread SCRIPT payloads? |
| `exp23_pack_script_r0.py` | Pack SCRIPT at strength R = 0 (0.05, 0.1, 0.2 bpnzac), first 5,000 images, |
| `exp24_second_source.py` | Second image source (journal roadmap WP5): does the picture change on other images? |
| `exp25_second_source_dctr.py` | DCTR + FLD on the second-source pair (BOSSbase-256, BOWS2-256). |
| `exp26_tuned_ensemble.py` | Does tuning the FLD ensemble change DCTR's numbers?  (journal roadmap) |
| `exp27_pack_robust_subset.py` | Pack a small stand-in for the 5,000-image grid dataset: covers plus GMAS, |
| `exp28_pack_second_source.py` | Pack the two 256x256 image sets (exp24) as one-file Kaggle datasets for SRNet. |
| `exp29_paper_tables.py` | Headline tables for the paper, built only from results files. |
| `exp30_paper_figures.py` | Figures for the full-scale results, built from exp29's paper_tables.json. |
| `exp31_srnet_warden_pooling.py` | Batch security with SRNet as the warden: SCRIPT at strength 0, and J-UNIWARD. |
| `exp32_script_footprint.py` | Why does SCRIPT's detectability at strength 0 hardly move with payload? |
| `exp33_script_by_texture.py` | Stage 2 of the SCRIPT flat-detectability question (see exp32): which images |
| `exp34_script_ablation.py` | Stage 3 of the SCRIPT flat-detectability question: which changes does the |
| `exp35_script_free_changes.py` | Stage 4 of the SCRIPT flat-detectability question: the zero-cost coefficients. |
| `exp36_si_reference.py` | Side-informed reference for SCRIPT: the same embedder without the selection. |
| `exp37_si_reference_dctr.py` | DCTR + FLD on the side-informed reference (exp36), beside SCRIPT at R = 0. |
| `exp38_pack_si_reference.py` | Pack the side-informed reference for the SRNet run on Kaggle: covers plus |
| `exp39_juniward_channel_pilot.py` | Pilot for the "J-UNIWARD plus error correction" baseline (F15). |
| `exp40_script_fix.py` | Can the fixed component of SCRIPT's detectability be removed? |
| `exp41_second_qf.py` | A second cover quality factor: does the picture hold at QF 75? |
| `exp42_pack_early.py` | Pack the curriculum dataset of exp42 as soon as its embedding is done, instead |
| `exp42_script_fix_curriculum.py` | Higher payloads of the repaired SCRIPT, so that a network can be given a fair |
| `exp43_si_reference_fix.py` | Side-informed reference for the repaired SCRIPT: is the selection still free? |
| `exp44_paired_permutation.py` | Is a P_E near chance a detection? A paired permutation test on saved scores. |
| `exp45_free_coefficient_modes.py` | Where are SCRIPT's zero-cost coefficients, and how far past the boundary? |
| `exp46_changes_at_zero_cost.py` | Which zero-cost coefficients does SCRIPT change, under each cost variant? |
| `exp48_script_released_patterns.py` | How much did a deviation in our SCRIPT harness change SCRIPT's results? |
| `exp49_random_subset.py` | Is the "first 5,000 images" subset representative? |
| `exp50_minicer_step_sweep.py` | MINICER: security and robustness against its QIM step. |
| `exp51_exact_dct_control.py` | Control for the zero-cost artefact: covers that agree with an exact DCT. |
| `exp53_qf75_harness_check.py` | Does the harness deviation of F36 matter at QF 75? |
| `exp54_si_fix_curriculum.py` | Dataset for SRNet on the like-for-like reference of the repaired SCRIPT. |
| `exp55_figure_data.py` | Data for three figures that no table of the paper already shows. |
| `make_timing_covers.py` | Generate JPEG covers for TIMING benchmarks only. |
| `prep_covers.py` | Convert BOSSbase PGMs into JPEG covers at a given quality factor. |

The `*_shard.m` files are the MATLAB embedding shards the Python runners call.
