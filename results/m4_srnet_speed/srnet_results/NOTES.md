# SRNet run notes

- Cells: GMAS_0.05, J-UNIWARD_0.2; seeds 0
- Hardware: 2 x Tesla T4; torch 2.11.0+cu128
- SRNet from scratch, 512x512 8-bit decoded pixels scaled to [0,1], pair constraint, 8 pairs per batch, Adamax 1e-3 then 1e-4 for the last fifth, L2 2e-4 on kernels, AMP, 40 epochs
- One paired 40/10/50 split per seed; epoch chosen on validation
- Null P_E: 0.4852 (chance 0.4875)
- Total minutes: 177
