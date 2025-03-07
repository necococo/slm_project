"""
モデル学習のためのトレーナークラス実装
WaveletモデルとTransformerの両方に対応
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any, List
from tqdm.auto import tqdm

class ModelTrainer:
    """
    Wavelet TransformerとベースラインTransformerの両方を
    学習できる汎用トレーナークラス
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        config: Any = None,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs",
    ):
        """
        Args:
            model: 学習対象のモデル
            train_dataset: 訓練データセット
            val_dataset: 検証データセット（オプション）
            test_dataset: テストデータセット（オプション）
            config: 学習設定
            device: 学習に使用するデバイス
            output_dir: 出力ディレクトリ
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # TensorBoardのセットアップ
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        
        # モデルをデバイスに移動
        self.model.to(self.device)
        
        # オプティマイザとスケジューラ
        self.setup_optimization()
        
        # AMP（自動混合精度）
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and self.device.type == "cuda" else None
        
        # メトリクス記録用
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.early_stop_count = 0
        
        self._log_start_info()
        
    def setup_optimization(self):
        """オプティマイザとスケジューラのセットアップ"""
        # AdamWオプティマイザ
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # 学習率スケジューラ（コサイン減衰）
        total_steps = len(self.train_dataset) * self.config.num_epochs // self.config.batch_size
        warmup_steps = int(total_steps * self.config.warmup_ratio) if self.config.warmup_ratio else self.config.warmup_steps
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=total_steps,  # 1サイクルの長さ
            T_mult=1,  # サイクル毎の乗数
            eta_min=1e-6  # 最小学習率
        )
        
    def _log_start_info(self):
        """学習開始時の情報ログ"""
        # モデルサイズ
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"モデルパラメータ数: {n_params:,} ({n_params/1e6:.2f}M)")
        
        # データセットサイズ
        print(f"訓練データセット: {len(self.train_dataset):,}サンプル")
        if self.val_dataset:
            print(f"検証データセット: {len(self.val_dataset):,}サンプル")
        if self.test_dataset:
            print(f"テストデセット: {len(self.test_dataset):,}サンプル")
            
        # 学習設定
        print(f"バッチサイズ: {self.config.batch_size}")
        print(f"初期学習率: {self.config.learning_rate}")
        print(f"エポック数: {self.config.num_epochs}")
        print(f"デバイス: {self.device}")
        
        # 設定保存
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            config_dict = {k: v for k, v in vars(self.config).items() if not k.startswith("_")}
            json.dump(config_dict, f, indent=2)
    
    def train(self, collate_fn=None):
        """
        モデルの学習を実行
        
        Args:
            collate_fn: DataLoaderで使用するcollate_function
        """
        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers if hasattr(self.config, "num_workers") else 2,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.eval_batch_size if hasattr(self.config, "eval_batch_size") else self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers if hasattr(self.config, "num_workers") else 2,
                pin_memory=True if self.device.type == "cuda" else False
            )
        else:
            val_loader = None
        
        # 学習ループ
        print(f"学習開始 - {self.config.num_epochs}エポック")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # 訓練
            train_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 検証
            if val_loader:
                val_loss = self._validate(val_loader, epoch)
                self.val_losses.append(val_loss)
                
                # モデル保存（ベスト）
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    self.early_stop_count = 0
                else:
                    self.early_stop_count += 1
                    
                # 早期停止
                if hasattr(self.config, "early_stop_patience") and \
                   self.config.early_stop_patience > 0 and \
                   self.early_stop_count >= self.config.early_stop_patience:
                    print(f"早期停止（{self.config.early_stop_patience}エポック改善なし）")
                    break
            
            # チェックポイント保存
            if (epoch + 1) % self.config.save_every == 0 or (epoch + 1) == self.config.num_epochs:
                self.save_checkpoint()
            
            # エポック終了ログ
            epoch_time = time.time() - epoch_start_time
            print(f"エポック {epoch+1}/{self.config.num_epochs} - "
                  f"訓練損失: {train_loss:.4f}, "
                  f"検証損失: {val_loss:.4f}, "
                  f"エポック時間: {epoch_time:.1f}秒")
            
        total_time = time.time() - start_time
        print(f"学習完了 - 合計時間: {total_time/60:.1f}分")
        
        # 最終モデル評価
        if self.test_dataset and hasattr(self.config, "eval_final_model") and self.config.eval_final_model:
            self._final_evaluation()
            
        # リソース解放
        self.writer.close()
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss
        }
    
    def _train_epoch(self, dataloader, epoch):
        """1エポックの学習を実行"""
        self.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"エポック {epoch+1}/{self.config.num_epochs} [訓練]")
        num_batches = len(dataloader)
        
        # 勾配蓄積のためのカウンタ
        accumulation_steps = self.config.gradient_accumulation_steps if hasattr(self.config, "gradient_accumulation_steps") else 1
        
        for batch_idx, batch in enumerate(progress_bar):
            # バッチデータをGPUに転送
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
            labels = batch["labels"].to(self.device) if "labels" in batch else None
            
            # 最初のステップで勾配をゼロ化
            if batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad()
                
            # 混合精度学習
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    
                    # 勾配蓄積のためのスケーリング
                    if accumulation_steps > 1:
                        loss = loss / accumulation_steps
                        
                    # 勾配計算
                    self.scaler.scale(loss).backward()
                    
                    # 勾配蓄積ステップの最後で最適化
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                        # 勾配クリッピング
                        if hasattr(self.config, "max_grad_norm") and self.config.max_grad_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            
                        # オプティマイザステップ
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # スケジューラステップ（バッチ単位で更新する場合）
                        if hasattr(self.config, "scheduler_step_by") and self.config.scheduler_step_by == "batch":
                            self.scheduler.step()
            else:
                # 通常の学習（混合精度なし）
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
                    
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # 勾配クリッピング
                    if hasattr(self.config, "max_grad_norm") and self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # スケジューラステップ
                    if hasattr(self.config, "scheduler_step_by") and self.config.scheduler_step_by == "batch":
                        self.scheduler.step()
            
            # ロスの累積（元のスケールに戻す）
            if accumulation_steps > 1:
                epoch_loss += loss.item() * accumulation_steps
            else:
                epoch_loss += loss.item()
            
            # プログレスバーの更新
            progress_bar.set_postfix(loss=loss.item())
            
            # TensorBoardにロギング（ステップ単位）
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
            self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            
        # エポック終了後のスケジューラステップ（エポック単位で更新する場合）
        if not hasattr(self.config, "scheduler_step_by") or self.config.scheduler_step_by == "epoch":
            self.scheduler.step()
        
        # 平均ロス計算
        avg_loss = epoch_loss / num_batches
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        
        return avg_loss
    
    def _validate(self, dataloader, epoch):
        """検証を実行"""
        self.model.eval()
        val_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"エポック {epoch+1}/{self.config.num_epochs} [検証]")
        
        with torch.no_grad():
            for batch in progress_bar:
                # バッチデータをGPUに転送
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device) if "labels" in batch else None
                
                # 混合精度評価
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                # ロス取得
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                val_loss += loss.item()
                
                # プログレスバーの更新
                progress_bar.set_postfix(loss=loss.item())
        
        # 平均ロス計算
        avg_loss = val_loss / len(dataloader)
        self.writer.add_scalar("validation/loss", avg_loss, epoch)
        
        return avg_loss
    
    def _final_evaluation(self):
        """最終的なモデル評価"""
        # ベストモデルを読み込み
        best_checkpoint_path = os.path.join(self.output_dir, "checkpoints", "best_model.pt")
        if os.path.exists(best_checkpoint_path):
            self.load_checkpoint(best_checkpoint_path)
            print("ベストモデルを読み込みました")
        
        # テストデータセットのデータローダー
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size if hasattr(self.config, "eval_batch_size") else self.config.batch_size,
            shuffle=False,
        )
        
        # テスト評価
        self.model.eval()
        test_loss = 0
        
        print("最終評価を実行中...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="テスト評価"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device) if "labels" in batch else None
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                test_loss += loss.item()
        
        # 平均テストロス
        avg_test_loss = test_loss / len(test_loader)
        print(f"最終テスト損失: {avg_test_loss:.4f}")
        
        # 結果をファイルに保存
        with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
            results = {
                "test_loss": avg_test_loss,
                "best_val_loss": self.best_val_loss,
                "train_loss": self.train_losses[-1] if self.train_losses else None,
            }
            json.dump(results, f, indent=2)
            
    def save_checkpoint(self, is_best=False):
        """チェックポイントを保存"""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        
        # 通常のチェックポイント
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        # 保存パス
        filename = "best_model.pt" if is_best else "last_checkpoint.pt"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # 保存
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            print(f"最良モデルを保存しました: {checkpoint_path} (検証損失: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """チェックポイントの読み込み"""
        # デバイス設定
        map_location = self.device.type
        
        # チェックポイント読み込み
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # モデルの状態復元
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # オプティマイザと学習率スケジューラの復元
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # その他の状態復元
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        
        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]
            
        print(f"チェックポイントを読み込みました: {checkpoint_path}")
