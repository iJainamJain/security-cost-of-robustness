# Scale run notes

- Dataset: 5000 covers, 18 cells
- Hardware: 2 x Tesla T4
- torch 2.11.0+cu128
- Part A: DCTR (Python port, validated against the MATLAB reference) + FLD ensemble, d_sub 800, L 200, 5 paired 50/50 splits
- Part B: EfficientNet-B0, ImageNet-pretrained, 1 channel, 512x512, no resize; 3 seeds, 40 epochs, batch 16, AdamW lr 1e-4, AMP; paired 40/10/50 split, epoch chosen on validation
- DCTR null P_E: NOT RUN
- CNN null P_E: 0.4901
- DCTR cells done: 0/18; CNN cells done: 6/18
- Total minutes: 420
