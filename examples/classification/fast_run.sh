#Fast run for smoke tests (short train/val/test loop)
python /Users/georg/workspace/PointNeXt/examples/classification/main.py \
  --cfg /Users/georg/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml \
  custom_dataset_root=/Users/georg/workspace/PointNet_KAN_Graphic/classification/handcrafted_dataset_v1 \
  fast_run=True fast_run_epochs=10 fast_run_train_batches=4 fast_run_val_batches=2 fast_run_test_batches=2
