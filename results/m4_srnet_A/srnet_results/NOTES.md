# SRNet curriculum, session A

- Hardware: 2 x Tesla T4; torch 2.11.0+cu128
- SRNet from scratch at the first stage of each chain, then fine-tuned down the payloads; 512x512 8-bit decoded pixels scaled to [0,1], pair constraint, 8 pairs per batch, Adamax 1e-3 then 1e-4 (after 80% of the epochs in the first stage, 60% in later stages), L2 2e-4 on kernels, AMP
- One paired 40/10/50 split per seed, shared by all stages of a chain; epoch chosen on validation, arriving weights included as a candidate
- Null test not repeated: passed in M4_srnet_speed (0.4852, chance 0.4875)
- Stages finished: 8 of 8
- Total minutes: 501
