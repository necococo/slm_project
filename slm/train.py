# slm/trainer.py
# Why not: 学習ループと評価を整理し、TensorBoard による監視を統合する

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Union
from cut_cross_entropy import linear_cross_entropy

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.diffusion import SimpleTextDiffusion
from slm.utils import get_model_size, compute_flops_per_batch
from slm.collator import CustomCollator


class Trainer:
    """
    How:
        Wave Networkモデルの訓練を実行するトレーナークラス。
        TensorBoardを使った学習過程のモニタリングと、複数の学習フェーズ管理を行います。
    """

    def __init__(
        self,
        model: WaveNetworkLM,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        training_config: Optional[TrainingConfig] = None,
        paths_config: Optional[PathsConfig] = None,
        device: torch.device = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.training_config = training_config or TrainingConfig()
        self.paths_config = paths_config or PathsConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルを通常のfloat32のままGPUへ (model.half()はしない)
        self.model.to(self.device)

        # ディレクトリ準備
        os.makedirs(self.paths_config.log_dir, exist_ok=True)
        os.makedirs(self.paths_config.checkpoint_dir, exist_ok=True)
        self.log_dir = self.paths_config.log_dir
        self.checkpoint_dir = self.paths_config.checkpoint_dir
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.paths_config.log_dir)
        
        # 学習率調整（NaN対策に小さめ）
        learning_rate = self.training_config.learning_rate
        # if learning_rate > 1e-5:
        #     print(f"WARNING: Lowering learning rate from {learning_rate} to 1e-5 for stability")
        #     learning_rate = 1e-5

        # Optimizer を AdamW に変更
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.training_config.weight_decay,
            eps=1e-5
        )
        
        # 追加: CosineAnnealingLR スケジューラの初期化
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.training_config.mlm_epochs,  # ここは必要に応じて調整
            eta_min=1e-6
        )
        
        # AMP用のGradScaler
        self.scaler = torch.amp.GradScaler('cuda') if (self.device.type == 'cuda' and self.training_config.use_amp) else None
        
        self._log_model_info()
        
    def _log_model_info(self):
        """モデルサイズなどをログ"""
        model_size = get_model_size(self.model)
        print(f"Model initialized with {model_size / 1e6:.2f}M parameters")
        print(f"Training on device: {self.device}")

    def train_mlm(self, num_epochs: Optional[int] = None) -> None:
        """
        MLM（Masked Language Model）方式で学習を実行
        """
        epochs = num_epochs or self.training_config.mlm_epochs
        
        tokenizer = self.model.config.tokenizer
        if tokenizer is None:
            raise ValueError("model.config.tokenizerが設定されていません。")

        # Collator
        collator = CustomCollator(
            tokenizer=tokenizer,
            model_config=self.model.config,
            mlm=True,
            mlm_probability=self.training_config.mlm_probability,
            mask_token_id=tokenizer.mask_token_id,
            qa=False
        )
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=collator
        )
        
        total_steps = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                step_loss = self._mlm_train_step(batch, total_steps)
                epoch_loss += step_loss
                total_steps += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {step_loss:.4f}")
            
            # エポック統計
            avg_loss = epoch_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            self.writer.add_scalar("MLM/Epoch Loss", avg_loss, epoch)
            self.writer.add_scalar("MLM/Epoch Time (s)", epoch_time, epoch)
            
            # エポック終了後にスケジューラを1ステップ進める
            self.scheduler.step()
            # 学習率を確認
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Current LR after scheduler step: {current_lr:.8f}")
            
            # Validation
            if self.valid_dataset:
                val_loss = self.validate()
                self.writer.add_scalar("MLM/Validation Loss", val_loss, epoch)
            
            # チェックポイント
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.save_checkpoint(f"mlm_epoch_{epoch+1}")

        total_time = time.time() - start_time
        self.writer.add_scalar("MLM/Total Training Time (min)", total_time / 60, 0)
        print(f"MLM training completed in {total_time / 60:.2f} minutes")

    def _mlm_train_step(self, batch: Dict[str, torch.Tensor], step: int) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        self.optimizer.zero_grad()

        # forward (mixed precision対応)
        # AMPを使いつつもlinear_cross_entropyのために明示的にfp16に変換
        embeddings = self.model(input_ids)
        classifier = self.model.get_classifier_weights()
        
        # 必ず半精度に変換（linear_cross_entropy の要件）
        embeddings = embeddings.half()
        classifier = classifier.half()
        
        # ロス計算
        loss = linear_cross_entropy(embeddings, classifier, labels)
        
        # 勾配計算
        loss.backward()
        
        # 勾配クリッピング（追加）
        if hasattr(self, 'clip_value') and self.clip_value:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
        
        self.optimizer.step()
            
        # ログ
        if step % 5 == 0:
            compute_time = 0.0  # 実測したい場合は time計測をどうぞ
            self.writer.add_scalar("MLM/Loss", loss.item(), step)
            self.writer.add_scalar("MLM/ComputeTime(ms)", compute_time * 1000, step)
            if torch.cuda.is_available():
                self.writer.add_scalar("System/GPU Memory (MB)", torch.cuda.memory_allocated()/1e6, step)
            est_flops = compute_flops_per_batch(self.model, input_ids.shape)
            self.writer.add_scalar("System/Estimated TFLOPS", est_flops/1e12/(compute_time+1e-9), step)
        
        return loss.item()

    def train_diffusion(self, num_epochs: Optional[int] = None) -> None:
        """
        拡散モデル方式でFine-tuning
        """
        epochs = num_epochs or self.training_config.diffusion_epochs
        if epochs == 0:
            print("Diffusion epochs = 0, skipping diffusion training")
            return

        vocab_size = self.model.get_classifier_weights().size(0)
        diffuser = SimpleTextDiffusion(timesteps=20, mask_token_id=4, vocab_size=vocab_size).to(self.device)
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True
        )
        
        total_steps = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                t = torch.randint(0, diffuser.timesteps, (1,)).item()
                step_loss = self._diffusion_train_step(batch, diffuser, t, total_steps)
                epoch_loss += step_loss
                total_steps += 1

                if batch_idx % 10 == 0:
                    print(f"Diffusion Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {step_loss:.4f}")

            avg_loss = epoch_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time
            print(f"Diffusion Epoch {epoch+1}/{epochs} done | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
            self.writer.add_scalar("Diffusion/Epoch Loss", avg_loss, epoch)
            self.writer.add_scalar("Diffusion/Epoch Time (s)", epoch_time, epoch)
            
            # エポック終了後にスケジューラを1ステップ進める
            self.scheduler.step()
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.save_checkpoint(f"diffusion_epoch_{epoch+1}")
        
        total_time = time.time() - start_time
        self.writer.add_scalar("Diffusion/Total Training Time (min)", total_time / 60, 0)
        print(f"Diffusion training done in {total_time / 60:.2f} minutes")

    def _diffusion_train_step(self, batch: Dict[str, torch.Tensor], diffuser: SimpleTextDiffusion, t: int, step: int) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                embeddings = self.model(input_ids)
                classifier = self.model.get_classifier_weights()
                # cut_cross_entropy は embeddings, classifier が fp16 である必要があるので
                embeddings = embeddings.half()
                classifier = classifier.half()
                loss = linear_cross_entropy(embeddings, classifier, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            embeddings = self.model(input_ids)
            classifier = self.model.get_classifier_weights()
            # cut_cross_entropy は embeddings, classifier が fp16 である必要があるので
            embeddings = embeddings.half()
            classifier = classifier.half()
            loss = linear_cross_entropy(embeddings, classifier, labels)  # 追加：lossの計算
            loss.backward()
            self.optimizer.step()
        
        if step % 5 == 0:
            self.writer.add_scalar("Diffusion/Loss", loss.item(), step)
            self.writer.add_scalar("Diffusion/Timestep", t, step)
        
        return loss.item()

    def validate(self) -> float:
        if not self.valid_dataset:
            return 0.0
        self.model.eval()
        
        # カスタムコレータの取得（これが欠けていた）
        tokenizer = self.model.config.tokenizer
        collator = CustomCollator(
            tokenizer=tokenizer,
            model_config=self.model.config,
            mlm=True,
            mlm_probability=self.training_config.mlm_probability,
            mask_token_id=tokenizer.mask_token_id,
            qa=False
        )
        
        dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=collator  # ここにコレータを追加
        )
        
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                embeddings = self.model(input_ids)
                classifier = self.model.get_classifier_weights()
                
                # 必ず半精度に変換（linear_cross_entropy の要件）
                embeddings = embeddings.half()
                classifier = classifier.half()
                
                loss = linear_cross_entropy(embeddings, classifier, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, name: str) -> None:
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model.config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {path}")

    def close(self) -> None:
        self.writer.close()
        print("TensorBoard writer closed and resources released")