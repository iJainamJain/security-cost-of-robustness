# M4 run notes

- Null test P_E: 0.4707 (chance at this test size is ~0.45, not 0.50)
- Hardware: Tesla T4
- torch 2.10.0+cu128, timm 1.0.26
- Detector: EfficientNet-B0, ImageNet-pretrained, in_chans=1, 512x512, no resize
- Epochs 40, batch 8, AdamW lr 1e-4, AMP; 5 seeds
- Paired split: 40% train / 10% validation / 50% test
- Augmentation: flips + 90-degree rotations only (no resampling)
- Epoch chosen on VALIDATION; P_E reported on the untouched test set

## Deviations from the spec
(none / describe)

## Anything that looked wrong
(describe, or 'nothing')
