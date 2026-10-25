# Chain run notes

- Chain: SCRIPT_0.05 -> SCRIPT_0.02 -> SCRIPT_0.01, epochs 40/25/25
- Hardware: 2 x Tesla T4; torch 2.11.0+cu128
- EfficientNet-B0, ImageNet-pretrained, 1 channel, 512x512, no resize; batch 16, AdamW lr 1e-4 (fresh optimiser per stage), AMP
- One paired 40/10/50 split per seed, shared by all stages; epoch chosen on validation, arriving weights included as a candidate
- CNN null P_E: 0.4901 (chance 0.4875)
- Seeds finished: [0, 1, 2, 3] of ['0', '1', '2', '3']
- Total minutes: 210
