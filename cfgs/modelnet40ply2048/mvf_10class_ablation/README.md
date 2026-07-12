# MVF 10-class ablation configs

All configs use seed `8240`, `deterministic: True`, the same explicit 10-class list, split ratios `[0.8, 0.1, 0.1]`, and the raw-frame roots from `cfgs/modelnet40ply2048/ultra-light-gen.yaml` unless the scenario is a baseline that requires a different input format.

Run from `PointNeXt` with:

```bash
python examples/classification/main.py --cfg cfgs/modelnet40ply2048/mvf_10class_ablation/<config>.yaml
```

Training 11 and 15 both use the existing learned residual correction head as the fine pose head. Training 15 sets `residual_correction_target: icp`, so ICP is used only as a detached teacher target; the forward path remains learned.

Config 20 enables pose-independent shared-anchor normalization. It centers every
view with the first view's centroid and scales all views with one joint radius.
Available GT poses supervise the registration head through a hybrid pose and
geometry loss, but samples without poses remain valid and inference requires no
pose metadata.

Config 21 implements the robust late-fusion result: each grouped view is
normalized and encoded independently with one shared PointNeXt, then object
logits are the mean of all valid view logits. It uses both object- and
view-level classification losses. A completely separate encoder, registration
head, and fine-ICP branch supplies geometry losses and fused-cloud exports but
is never used to calculate classification logits; no pose input is required.
The geometry branch receives a second tensor normalized with the first view's
centroid and one joint radius over all views, so exported clouds preserve a
common scale while classifier views remain independently normalized.

Config 22 tests the efficient one-encoder alternative. All object views use the
same first-view centroid and joint radius, one PointNeXt pass supplies per-view
classification features, and mean logits produce the class prediction. Detached
copies of those features feed a registration head and fine ICP, so geometry
losses cannot update the shared classification encoder.

Config 23 is identical to Config 22 except that geometry features are not
detached. Classification, alignment, and ICP regularization therefore jointly
train the one shared PointNeXt encoder; this isolates the effect of multi-task
geometry gradients.
