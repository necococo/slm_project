"""
Wave Network実装のデバッグツール
ノードやトレーニングのボトルネックを特定します
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

class WaveDebugger:
    """
    Wave Network特有の問題をデバッグするためのユーティリティ
    """
    
    @staticmethod
    def check_wave_representation(model, sample_input, save_path=None):
        """
        波表現の統計解析を行い、数値的に健全かチェック
        """
        model.eval()
        stats = {}
        hooks = []
        wave_repr_data = {}
        
        # 波表現をフックで捕捉する関数
        def get_wave_hook(name, is_global=False):
            def hook(module, input, output):
                wave_repr_data[name] = {
                    'real': output[0].detach().cpu().numpy(),
                    'imag': output[1].detach().cpu().numpy(),
                    'global': is_global
                }
            return hook
        
        # 各レイヤーにフックを追加
        for i, layer in enumerate(model.layers):
            # コンピュートメソッドにフックを追加
            compute_wave_fn = layer.wave_layer.__module__.compute_wave_representation
            hooks.append(
                compute_wave_fn.register_forward_hook(
                    get_wave_hook(f"layer_{i}_token_wave", False)
                )
            )
            
        # 推論実行
        with torch.no_grad():
            _ = model(sample_input)
        
        # 統計解析
        for name, data in wave_repr_data.items():
            real, imag = data['real'], data['imag']
            stats[name] = {
                'real_mean': np.mean(real),
                'real_std': np.std(real),
                'real_min': np.min(real),
                'real_max': np.max(real),
                'imag_mean': np.mean(imag),
                'imag_std': np.std(imag),
                'imag_min': np.min(imag),
                'imag_max': np.max(imag),
                'magnitude_mean': np.mean(np.sqrt(real**2 + imag**2)),
                'has_nan_real': np.isnan(real).any(),
                'has_nan_imag': np.isnan(imag).any(),
                'has_inf_real': np.isinf(real).any(),
                'has_inf_imag': np.isinf(imag).any(),
            }
        
        # フックを削除
        for hook in hooks:
            hook.remove()
        
        # 可視化（オプション）
        if save_path:
            fig, axes = plt.subplots(len(wave_repr_data), 2, figsize=(12, 4*len(wave_repr_data)))
            for i, (name, data) in enumerate(wave_repr_data.items()):
                real, imag = data['real'], data['imag']
                # 実部のヒストグラム
                axes[i, 0].hist(real.flatten(), bins=50)
                axes[i, 0].set_title(f"{name} - Real Part")
                # 虚部のヒストグラム
                axes[i, 1].hist(imag.flatten(), bins=50)
                axes[i, 1].set_title(f"{name} - Imaginary Part")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return stats
    
    @staticmethod
    def analyze_gradients(model, optimizer, dataloader, trainer, num_steps=5, clip_value=None):
        """
        勾配の統計を解析し、学習障害の原因を特定
        """
        gradient_stats = []
        model.train()
        
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break
                
            # 勾配リセット
            optimizer.zero_grad()
            
            # 通常の順伝播と逆伝播
            input_ids = batch["input_ids"].to(next(model.parameters()).device)
            labels = batch["labels"].to(next(model.parameters()).device)
            
            # forward
            embeddings = model(input_ids)
            classifier = model.get_classifier_weights()
            # cut_cross_entropy は embeddings, classifier が fp16 である必要があるので
            embeddings = embeddings.half()
            classifier = classifier.half()
            
            # ロス計算
            from cut_cross_entropy import linear_cross_entropy
            loss = linear_cross_entropy(embeddings, classifier, labels)
            loss.backward()
            
            # 勾配の解析
            batch_stats = {
                'loss': loss.item(),
                'layer_gradients': {},
                'overall_grad_norm': 0.0
            }
            
            overall_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    overall_grad_norm += grad_norm ** 2
                    
                    # 特に大きな勾配や小さすぎる勾配をチェック
                    if grad_norm > 10.0 or grad_norm < 1e-6:
                        batch_stats['layer_gradients'][name] = {
                            'norm': grad_norm,
                            'mean': param.grad.mean().item(),
                            'std': param.grad.std().item() if param.grad.numel() > 1 else 0,
                            'has_nan': torch.isnan(param.grad).any().item(),
                            'has_inf': torch.isinf(param.grad).any().item()
                        }
            
            batch_stats['overall_grad_norm'] = np.sqrt(overall_grad_norm)
            gradient_stats.append(batch_stats)
            
            # 勾配クリッピング（指定されている場合）
            if clip_value:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
            optimizer.step()
        
        return gradient_stats
