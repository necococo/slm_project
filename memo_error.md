エポック 5 の検証中...
Validation:   0% 0/10 [00:00<?, ?it/s]/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [68,0,0], thread: [64,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [68,0,0], thread: [65,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [68,0,0], thread: [66,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/


/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [66,0,0], thread: [127,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
WARNING: Error during validation: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

検証損失: 100.0000
トレーニング中にエラーが発生しました: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Traceback (most recent call last):
  File "/content/slm_project/slm/main.py", line 1102, in main
    trainer.train_diffusion(start_epoch=start_epoch, start_step=start_step)
  File "/content/slm_project/slm/train.py", line 548, in train_diffusion
    self.save_checkpoint(checkpoint_name)
  File "/content/slm_project/slm/train.py", line 786, in save_checkpoint
    torch.save(checkpoint, checkpoint_path)
  File "/usr/local/lib/python3.11/dist-packages/torch/serialization.py", line 944, in save
    _save(
  File "/usr/local/lib/python3.11/dist-packages/torch/serialization.py", line 1214, in _save
    storage = storage.cpu()
              ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/storage.py", line 267, in cpu
    return torch.UntypedStorage(self.size()).copy_(self, False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

