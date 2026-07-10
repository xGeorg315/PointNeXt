# MVF 10-class ablation configs

All configs use seed `8240`, `deterministic: True`, the same explicit 10-class list, split ratios `[0.8, 0.1, 0.1]`, and the raw-frame roots from `cfgs/modelnet40ply2048/ultra-light-gen.yaml` unless the scenario is a baseline that requires a different input format.

Run from `PointNeXt` with:

```bash
python examples/classification/main.py --cfg cfgs/modelnet40ply2048/mvf_10class_ablation/<config>.yaml
```

Training 11 and 15 both use the existing learned residual correction head as the fine pose head. Training 15 sets `residual_correction_target: icp`, so ICP is used only as a detached teacher target; the forward path remains learned.
