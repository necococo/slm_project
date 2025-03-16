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

---------
エポック 5 の検証中...
Validation:   0% 0/10 [00:00<?, ?it/s]バリデーション開始 (最大10バッチ)
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [54,0,0], thread: [32,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [54,0,0], thread: [33,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [54,0,0], thread: [34,0,0] Assertion `srcIndex < srcSelectDimSize` failed.

/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1422: indexSelectLargeIndex: block: [270,0,0], thread: [127,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
バリデーションのバッチ 0 でエラー発生: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Traceback (most recent call last):
  File "/content/slm_project/slm/train.py", line 724, in validate
    embeddings = self.model(input_ids)
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 819, in forward
    return model_forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 807, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/content/slm_project/slm/modules/wave_network.py", line 548, in forward
    hidden_states = layer(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/slm_project/slm/modules/wave_network.py", line 404, in forward
    wave_output = self.wave_layer(wave_input)  # [B, S, D]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/slm_project/slm/modules/wave_network.py", line 265, in forward
    if torch.isnan(x).any():
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

検証損失: 100.0000
CUDAエラーが発生しましたが、パラメータを一つずつCPUに移動して再試行します: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

警告: パラメータ token_embedding.weight のCPU移動に失敗しました。スキップします。
警告: パラメータ layers.0.wave_layer.ffn_real.0.weight のCPU移動に失敗しました。スキップします。

警告: オプティマイザのテンソル exp_avg_sq のCPU移動に失敗しました。スキップします。
Checkpoint saved to /content/drive/MyDrive/slm_outputs/slm_1024h_3l/checkpoints/final_model.pt
Diffusion Epoch 6/104:   0% 0/32 [00:00<?, ?it/s]
トレーニング中にエラーが発生しました: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Traceback (most recent call last):
  File "/content/slm_project/slm/main.py", line 1102, in main
    trainer.train_diffusion(start_epoch=start_epoch, start_step=start_step)
  File "/content/slm_project/slm/train.py", line 433, in train_diffusion
    for batch_idx, batch in enumerate(progress_bar):
  File "/usr/local/lib/python3.11/dist-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/usr/local/lib/python3.11/dist-packages/accelerate/data_loader.py", line 572, in __iter__
    current_batch = send_to_device(current_batch, self.device, non_blocking=self._non_blocking)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 183, in send_to_device
    {
  File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 184, in <dictcomp>
    k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 155, in send_to_device
    return tensor.to(device, non_blocking=non_blocking)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

]



ローカルデータセットを使用します
スプリット別ディレクトリ形式のデータセットを読み込み中...
開発用に訓練データを制限: 732655 → 1000 件

モデルを初期化します...
バッチサイズを調整: 128 → 32 (速度と安定性のため)
学習率を調整: 2e-05 → 5e-06 (安定性のため)



特殊トークン情報:
  MASK: <mask> (ID: 32100)
  語彙サイズ: 32000