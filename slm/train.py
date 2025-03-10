# slm/train.py
# Why not: 学習ループと評価を整理し、TensorBoard による監視を統合する

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F  # F.cross_entropy用に追加
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Union, Any, List
from cut_cross_entropy import linear_cross_entropy

# Accelerateをインポート
from accelerate import Accelerator
from accelerate.utils import set_seed

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
        device: torch.device = None,
        seed: int = 42
    ):
        # ランダムシードを設定して再現性を確保
        set_seed(seed)
        
        # Acceleratorの初期化 - TPU v5e-1向け設定
        self.accelerator = Accelerator(
            mixed_precision="bf16",  # TPUではbf16が最適
            gradient_accumulation_steps=1,
            log_with=None,  # TensorBoardを自分で管理
            project_config=None
        )
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.training_config = training_config or TrainingConfig()
        self.paths_config = paths_config or PathsConfig()
        
        # deviceはAcceleratorが管理するため、直接指定しなくてOK
        self.device = self.accelerator.device
        print(f"Acceleratorデバイス: {self.device}")
        
        self.model = model
        
        # ディレクトリ準備
        os.makedirs(self.paths_config.log_dir, exist_ok=True)
        os.makedirs(self.paths_config.checkpoint_dir, exist_ok=True)
        self.log_dir = self.paths_config.log_dir
        self.checkpoint_dir = self.paths_config.checkpoint_dir
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.paths_config.log_dir)
        
        # 学習率調整
        learning_rate = self.training_config.learning_rate

        # Optimizer を AdamW に変更
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.training_config.weight_decay,
            eps=1e-5
        )
        
        # スケジューラの初期化
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max(self.training_config.mlm_epochs, self.training_config.diffusion_epochs, 1),
            eta_min=1e-6
        )
        
        # AMPはAcceleratorが管理するためscalerは不要
        self.scaler = None
        
        # Acceleratorによるモデルとオプティマイザの準備
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        
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
        
        # tqdmのインポートを確保
        try:
            from tqdm.auto import tqdm
        except ImportError:
            # tqdmがない場合はダミーのtqdmを作成
            def tqdm(iterable, **kwargs):
                return iterable
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            # tqdmでラップしてプログレスバーを表示
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"MLM Epoch {epoch+1}/{epochs}", leave=False)):
                step_loss = self._mlm_train_step(batch, total_steps)
                epoch_loss += step_loss
                total_steps += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Training Batch {batch_idx}/{len(dataloader)} | Loss: {step_loss:.4f}")
            
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
        
        # メモリ最適化: バッチサイズが大きすぎる場合の対策
        batch_size = input_ids.size(0)
        max_micro_batch = 4  # マイクロバッチの最大サイズ (調整可能)
        
        # メモリ使用量を減らすためにCPUキャッシュをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.optimizer.zero_grad()
        
        # マイクロバッチでの処理（大きなバッチを小さく分割）
        if batch_size > max_micro_batch and self.training_config.use_amp:
            # バッチを分割
            num_micro_batches = (batch_size + max_micro_batch - 1) // max_micro_batch
            total_loss = 0.0
            
            # 入力と出力のチェック
            print(f"入力形状: {input_ids.shape}, ラベル形状: {labels.shape}")
            
            for i in range(num_micro_batches):
                start_idx = i * max_micro_batch
                end_idx = min((i + 1) * max_micro_batch, batch_size)
                micro_input_ids = input_ids[start_idx:end_idx]
                micro_labels = labels[start_idx:end_idx]
                
                # オリジナルと同じ損失計算ロジック（マイクロバッチに対して）
                if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model(micro_input_ids)
                        classifier = self.model.get_classifier_weights()
                        embeddings = embeddings.half()
                        classifier = classifier.half()
                        micro_loss = linear_cross_entropy(embeddings, classifier, micro_labels)
                else:
                    with torch.cuda.amp.autocast():
                        logits = self.model(micro_input_ids)
                        micro_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), micro_labels.view(-1))
                
                # 損失を正規化してバッチサイズ不変でスケール
                micro_loss = micro_loss * (end_idx - start_idx) / batch_size
                
                # 勾配を蓄積
                micro_loss.backward()
                total_loss += micro_loss.item() * batch_size / (end_idx - start_idx)
                
                # マイクロバッチごとにメモリを解放
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            loss_item = total_loss
        else:
            # 通常の処理（バッチサイズが小さい場合）
            if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
                # Cut Cross Entropy用の処理
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        embeddings = self.model(input_ids)  # hidden_statesが返される
                        classifier = self.model.get_classifier_weights()
                        # 必ず半精度に変換（linear_cross_entropy の要件）
                        embeddings = embeddings.half()
                        classifier = classifier.half()
                        loss = linear_cross_entropy(embeddings, classifier, labels)
                else:
                    embeddings = self.model(input_ids)
                    classifier = self.model.get_classifier_weights()
                    loss = linear_cross_entropy(embeddings, classifier, labels)
                    
                    # デバッグ用：分類器の勾配情報をランダムに表示（1%の確率）
                    if random.random() < 0.01:
                        print(f"\n===== 勾配計算前（ステップ {step}）=====")
                        print(f"分類器の重みの形状: {classifier.shape}")
                        print(f"分類器のrequires_grad: {classifier.requires_grad}")
            else:
                # 通常のCross Entropy用の処理
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids)  # logitsが返される
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            # 勾配計算
            loss.backward()
            loss_item = loss.item()
            
            # デバッグ用：勾配計算後に分類器の勾配情報を表示（1%の確率）
            if random.random() < 0.01 and hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
                print(f"\n===== 勾配計算後（ステップ {step}）=====")
                classifier_weight = self.model.classifier.weight
                if hasattr(classifier_weight, 'grad'):
                    grad_status = "存在する" if classifier_weight.grad is not None else "None"
                    if classifier_weight.grad is not None:
                        print(f"分類器の勾配ノルム: {classifier_weight.grad.norm().item():.6f}")
                        print(f"分類器の勾配最大値: {classifier_weight.grad.max().item():.6f}")
                        print(f"分類器の勾配最小値: {classifier_weight.grad.min().item():.6f}")
                    else:
                        print(f"分類器の勾配状態: {grad_status}")
                else:
                    print("分類器の勾配属性が存在しません")
            
        # 勾配クリッピング（追加）
        if hasattr(self.training_config, 'clip_value') and self.training_config.clip_value:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.clip_value)
        
        self.optimizer.step()
            
        # ログ
        if step % 5 == 0:
            compute_time = 0.0  # 実測したい場合は time計測をどうぞ
            self.writer.add_scalar("MLM/Loss", loss_item, step)
            self.writer.add_scalar("MLM/ComputeTime(ms)", compute_time * 1000, step)
            if torch.cuda.is_available():
                self.writer.add_scalar("System/GPU Memory (MB)", torch.cuda.memory_allocated()/1e6, step)
                self.writer.add_scalar("System/GPU Memory Free (MB)", torch.cuda.get_device_properties(0).total_memory/1e6 - torch.cuda.memory_allocated()/1e6, step)
            est_flops = compute_flops_per_batch(self.model, input_ids.shape)
            self.writer.add_scalar("System/Estimated TFLOPS", est_flops/1e12/(compute_time+1e-9), step)
        
        return loss_item

    def train_diffusion(self, num_epochs: Optional[int] = None) -> None:
        """
        拡散モデル方式でFine-tuning - TPU v5e-1 + Accelerateを使用
        """
        epochs = num_epochs or self.training_config.diffusion_epochs
        if epochs == 0:
            print("Diffusion epochs = 0, skipping diffusion training")
            return

        vocab_size = self.model.get_classifier_weights().size(0)
        # マスクトークンIDをトークナイザーから取得（初期値は4）
        mask_token_id = 4  # デフォルト値
        if hasattr(self.model.config, 'tokenizer') and self.model.config.tokenizer is not None:
            if hasattr(self.model.config.tokenizer, 'mask_token_id'):
                mask_token_id = self.model.config.tokenizer.mask_token_id
            
        # diffuserもacceleratorで準備
        diffuser = SimpleTextDiffusion(
            timesteps=20, 
            mask_token_id=mask_token_id, 
            vocab_size=vocab_size
        )
        diffuser = self.accelerator.prepare(diffuser)
        
        # diffusion学習用のカスタムコレーターを使用
        try:
            from slm.collator import CustomCollator
            # トークナイザーの取得
            tokenizer = self.model.config.tokenizer
            if tokenizer is None:
                print("警告: tokenizer が None です。デフォルトのコレーターを使用します。")
                collator = None
            else:
                # diffusion学習用のカスタムコレーター - マスキングはDiffusionが行うので無効化
                collator = CustomCollator(
                    tokenizer=tokenizer,
                    model_config=self.model.config,
                    mlm=False,  # マスキングを無効化（Diffusionが行う）
                    mlm_probability=0.0,  # マスキング確率を0に
                    mask_token_id=mask_token_id,
                    qa=False
                )
                print(f"Diffusion学習用のカスタムコレーターを初期化しました。コレーターのマスキングは無効です。")
        except Exception as e:
            print(f"コレーターの初期化中にエラーが発生しました: {e}")
            collator = None
        
        # 固定バッチサイズ8を使用
        batch_size = 8
        print(f"Diffusion学習のバッチサイズを{batch_size}に固定")
        
        # DataLoaderを作成し、Acceleratorで準備
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=4,  # TPUではマルチプロセスデータロードが効果的
            pin_memory=True  # データ転送を高速化
        )
        dataloader = self.accelerator.prepare(dataloader)
        
        total_steps = 0
        start_time = time.time()
        
        # Accelerate組み込みのプログレスバーを使用
        from tqdm.auto import tqdm
        
        # Accelerator用のマスターランク確認（メインプロセスのみ出力）
        is_main_process = self.accelerator.is_main_process
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            # tqdmはAcceleratorと組み合わせて使用
            progress_bar = tqdm(
                dataloader, 
                desc=f"Diffusion Epoch {epoch+1}/{epochs}", 
                disable=not is_main_process
            )
            
            self.model.train()
            for batch_idx, batch in enumerate(progress_bar):
                # タイムステップをランダムに選択
                t = torch.randint(0, diffuser.timesteps, (1,)).item()
                
                # Acceleratorを使用した学習ステップ
                with self.accelerator.accumulate(self.model):
                    step_loss = self._diffusion_train_step_accelerate(batch, diffuser, t, total_steps)
                    epoch_loss += step_loss
                    
                    # プログレスバーに現在の損失を表示
                    progress_bar.set_postfix(loss=f"{step_loss:.4f}")

                total_steps += 1

            # エポック統計（メインプロセスのみ）
            if is_main_process:
                avg_loss = epoch_loss / len(dataloader)
                epoch_time = time.time() - epoch_start_time
                print(f"Diffusion Epoch {epoch+1}/{epochs} done | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
                self.writer.add_scalar("Diffusion/Epoch Loss", avg_loss, epoch)
                self.writer.add_scalar("Diffusion/Epoch Time (s)", epoch_time, epoch)
            
            # スケジューラを進める
            self.scheduler.step()
            
            # チェックポイント保存（メインプロセスのみ）
            if is_main_process and ((epoch + 1) % 5 == 0 or epoch == epochs - 1):
                self.save_checkpoint(f"diffusion_epoch_{epoch+1}")
        
        # 学習完了（メインプロセスのみ）
        if is_main_process:
            total_time = time.time() - start_time
            self.writer.add_scalar("Diffusion/Total Training Time (min)", total_time / 60, 0)
            print(f"Diffusion training done in {total_time / 60:.2f} minutes")

    def _diffusion_train_step(self, batch: Dict[str, Any], diffuser: SimpleTextDiffusion, t: int, step: int) -> float:
        self.model.train()
        
        # バッチの変換を安全に行う
        def safe_to_device(tensor_or_list):
            if isinstance(tensor_or_list, torch.Tensor):
                return tensor_or_list.to(self.device)
            elif isinstance(tensor_or_list, list):
                # リストならテンソルに変換
                return torch.tensor(tensor_or_list, device=self.device)
            else:
                raise ValueError(f"Unsupported type: {type(tensor_or_list)}")
        
        # input_idsとlabelsキーが存在するか確認（データフォーマットが異なる可能性に対応）
        if "input_ids" in batch:
            input_ids = safe_to_device(batch["input_ids"])
        elif "tokens" in batch:
            input_ids = safe_to_device(batch["tokens"])
        else:
            raise ValueError("バッチにinput_idsまたはtokensキーが存在しません")
            
        # ラベルの処理
        if "labels" in batch:
            labels = safe_to_device(batch["labels"])
        else:
            # ラベルが提供されていない場合は、diffuserを使って生成する
            noisy_tokens, labels = diffuser(input_ids.clone(), torch.tensor([t], device=self.device))
        
        self.optimizer.zero_grad()
        
        # use_cut_cross_entropyフラグに基づいて処理を変更
        if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
            # Cut Cross Entropy用の処理
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
                embeddings = embeddings.half()
                classifier = classifier.half()
                loss = linear_cross_entropy(embeddings, classifier, labels)
                loss.backward()
                self.optimizer.step()
        else:
            # 通常のCross Entropy用の処理
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
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
        
        # メモリを解放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 検証用のバッチサイズは小さめに設定（OOM対策）
        eval_batch_size = min(4, self.training_config.batch_size)
        
        # カスタムコレータの取得
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
            batch_size=eval_batch_size,  # 小さめのバッチサイズを使用
            shuffle=False,
            collate_fn=collator
        )
        
        # 検証中のメモリを監視（オプション）
        max_memory_used = 0
        if torch.cuda.is_available():
            max_memory_used = torch.cuda.memory_allocated() / 1e6
        
        total_loss = 0.0
        total_batches = min(10, len(dataloader))  # 最大10バッチに制限
        
        # tqdmのインポートを確保
        try:
            from tqdm.auto import tqdm
        except ImportError:
            # tqdmがない場合はダミーのtqdmを作成
            def tqdm(iterable, **kwargs):
                return iterable
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Validation", total=total_batches, leave=False)):
                # 最大10バッチまでで終了（メモリ使用量を制限）
                if i >= total_batches:
                    break
                    
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                try:
                    # AMPを使用
                    if self.training_config.use_amp and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            # use_cut_cross_entropyフラグに基づいて処理を変更
                            if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
                                # Cut Cross Entropy用の処理
                                embeddings = self.model(input_ids)
                                classifier = self.model.get_classifier_weights()
                                # 必ず半精度に変換（linear_cross_entropy の要件）
                                embeddings = embeddings.half()
                                classifier = classifier.half()
                                loss = linear_cross_entropy(embeddings, classifier, labels)
                            else:
                                # 通常のCross Entropy用の処理
                                logits = self.model(input_ids)
                                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        # AMPなしの処理
                        if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
                            # Cut Cross Entropy用の処理
                            embeddings = self.model(input_ids)
                            classifier = self.model.get_classifier_weights()
                            # 必ず半精度に変換（linear_cross_entropy の要件）
                            embeddings = embeddings.half()
                            classifier = classifier.half()
                            loss = linear_cross_entropy(embeddings, classifier, labels)
                        else:
                            # 通常のCross Entropy用の処理
                            logits = self.model(input_ids)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    total_loss += loss.item()
                    
                    # メモリ使用量を監視
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated() / 1e6
                        max_memory_used = max(max_memory_used, current_memory)
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: GPU ran out of memory during validation, using loss from {i} batches")
                        if i == 0:
                            # 最初のバッチでOOMが発生した場合、高い損失を返す
                            return 100.0
                        break
                    else:
                        raise e
                
                # バッチごとにメモリを解放
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 処理したバッチ数で割る
        processed_batches = min(i + 1, total_batches)
        avg_loss = total_loss / processed_batches if processed_batches > 0 else 100.0
        
        print(f"Validation Loss: {avg_loss:.4f} [Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.2f}] (using {processed_batches} batches, max memory: {max_memory_used:.2f}MB)")
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