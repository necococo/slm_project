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
        Acceleratorを使用してTPU v5e-1に最適化
        """
        epochs = num_epochs or self.training_config.mlm_epochs
        
        # MLMエポックが0の場合はスキップ
        if epochs == 0:
            print("MLM epochs = 0, skipping MLM training")
            return
        
        tokenizer = self.model.config.tokenizer
        if tokenizer is None:
            raise ValueError("model.config.tokenizerが設定されていません。")

        # マスターランク確認（メインプロセスのみ出力）
        is_main_process = self.accelerator.is_main_process
        
        # Collator
        collator = CustomCollator(
            tokenizer=tokenizer,
            model_config=self.model.config,
            mlm=True,
            mlm_probability=self.training_config.mlm_probability,
            mask_token_id=tokenizer.mask_token_id,
            qa=False
        )
        
        # 固定バッチサイズ8を使用（TPU v5e-1向け）
        batch_size = 8
        if is_main_process:
            print(f"MLM学習のバッチサイズを{batch_size}に固定")
        
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
        
        # tqdmによるプログレスバー
        from tqdm.auto import tqdm
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            # tqdmはメインプロセスでのみ表示
            progress_bar = tqdm(
                dataloader, 
                desc=f"MLM Epoch {epoch+1}/{epochs}", 
                disable=not is_main_process
            )
            
            self.model.train()
            for batch_idx, batch in enumerate(progress_bar):
                # Acceleratorを使用した学習ステップ
                with self.accelerator.accumulate(self.model):
                    # 新しいTPU対応学習ステップを呼び出す
                    step_loss = self._mlm_train_step_accelerate(batch, total_steps)
                    epoch_loss += step_loss
                    
                    # プログレスバーに現在の損失を表示
                    progress_bar.set_postfix(loss=f"{step_loss:.4f}")
                    
                total_steps += 1
            
            # エポック統計（メインプロセスのみ）
            if is_main_process:
                avg_loss = epoch_loss / len(dataloader)
                epoch_time = time.time() - epoch_start_time
                print(f"MLM Epoch {epoch+1}/{epochs} done | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
                self.writer.add_scalar("MLM/Epoch Loss", avg_loss, epoch)
                self.writer.add_scalar("MLM/Epoch Time (s)", epoch_time, epoch)
            
            # スケジューラを進める
            self.scheduler.step()
            
            # 学習率を確認（メインプロセスのみ）
            if is_main_process:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Current LR after scheduler step: {current_lr:.8f}")
            
            # Validation
            if self.valid_dataset:
                val_loss = self.validate()
                if is_main_process:
                    self.writer.add_scalar("MLM/Validation Loss", val_loss, epoch)
            
            # チェックポイント保存（メインプロセスのみ）
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.save_checkpoint(f"mlm_epoch_{epoch+1}")

        # 学習完了ログ（メインプロセスのみ）
        if is_main_process:
            total_time = time.time() - start_time
            self.writer.add_scalar("MLM/Total Training Time (min)", total_time / 60, 0)
            print(f"MLM training completed in {total_time / 60:.2f} minutes")

    def _mlm_train_step_accelerate(self, batch: Dict[str, torch.Tensor], step: int) -> float:
        """
        Acceleratorを使用したMLM学習ステップ
        TPU v5e-1向けに最適化
        """
        self.model.train()
        
        # 入力データ - Acceleratorが既にデバイスに配置済み
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # オプティマイザのゼロ勾配（Acceleratorが管理）
        self.optimizer.zero_grad()
        
        # use_cut_cross_entropyフラグに基づいて処理を変更
        if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
            # Cut Cross Entropy用の処理 - TPU v5e-1向けにbfloat16を使用
            with self.accelerator.autocast():
                embeddings = self.model(input_ids)
                classifier = self.model.get_classifier_weights()
                # TPUではbfloat16に変換
                if hasattr(torch, 'bfloat16'):
                    embeddings = embeddings.to(torch.bfloat16)
                    classifier = classifier.to(torch.bfloat16)
                else:
                    # bfloat16がない場合はfloat16を使用
                    embeddings = embeddings.half()
                    classifier = classifier.half()
                loss = linear_cross_entropy(embeddings, classifier, labels)
        else:
            # 通常のCross Entropy用の処理
            with self.accelerator.autocast():
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Acceleratorによる後方伝播
        self.accelerator.backward(loss)
        
        # 勾配クリッピング（設定されている場合）
        if hasattr(self.training_config, 'clip_value') and self.training_config.clip_value:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_config.clip_value)
        
        # オプティマイザのステップ
        self.optimizer.step()
        
        # ロギング（メインプロセスのみ）
        if self.accelerator.is_main_process and step % 5 == 0:
            # 損失値を集約
            loss_value = self.accelerator.gather(loss).mean().item()
            self.writer.add_scalar("MLM/Loss", loss_value, step)
            
            # TPU利用率などの代わりにモデルのFLOPsなど計算量の見積もりをロギング
            est_flops = compute_flops_per_batch(self.model, input_ids.shape)
            self.writer.add_scalar("System/Estimated TFLOPS", est_flops/1e12, step)
        
        # 損失値を返す前に集約（コミュニケーションコスト削減のため条件付き）
        if step % 5 == 0:
            return self.accelerator.gather(loss).mean().item()
        else:
            return loss.item()  # ローカルな損失値のみを返す

    def train_diffusion(self, num_epochs: Optional[int] = None, start_epoch: int = 0) -> None:
        """
        拡散モデル方式でFine-tuning - TPU v5e-1 + Accelerateを使用
        start_epoch: 開始エポック番号（チェックポイントから再開する場合に使用）
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
            timesteps=10,  # タイムステップを減らして計算効率を向上
            mask_token_id=mask_token_id, 
            vocab_size=vocab_size,
            beta_schedule="quadratic"  # より安定した学習のため二次関数的なスケジュールを使用
        )
        # diffuserをクラス属性として保存（データ確認時などに参照できるように）
        self.diffuser = diffuser
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
        
        # 設定からバッチサイズを取得
        batch_size = self.training_config.batch_size
        print(f"Diffusion学習のバッチサイズ: {batch_size}")
        
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
        
        # 学習開始前にトークナイザーとデータの簡単な確認（最初のエポックのみ）
        if is_main_process:
            try:
                print("\n=== 学習開始前のトークナイザーとデータ確認 ===")
                # DataLoaderから最初のバッチを取得
                sample_batch = next(iter(dataloader))
                
                # モデルのトークナイザー情報を取得
                tokenizer = self.model.config.tokenizer if hasattr(self.model.config, 'tokenizer') else None
                
                if tokenizer is not None:
                    # マスクトークンの確認
                    print(f"マスクトークン: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
                    
                    # バッチの形状確認
                    print(f"バッチの形状: input_ids={sample_batch['input_ids'].shape}, "
                          f"attention_mask={sample_batch['attention_mask'].shape if 'attention_mask' in sample_batch else 'なし'}, "
                          f"labels={sample_batch['labels'].shape if 'labels' in sample_batch else 'なし'}")
                    
                    # サンプルデータの表示（最初の例）
                    if 'input_ids' in sample_batch:
                        sample_ids = sample_batch['input_ids'][0].cpu().tolist()
                        print(f"サンプルのトークンID (最初の20個): {sample_ids[:20]}")
                        
                        # トークンIDをデコードして表示
                        sample_text = tokenizer.decode(sample_ids)
                        print(f"サンプルテキスト: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
                        
                        # Diffusionモデルのノイズ追加テスト
                        if hasattr(self, 'diffuser'):
                            print("\nDiffusionノイズ追加テスト:")
                            # 最大ノイズレベル（タイムステップ）で
                            t = torch.tensor([self.diffuser.timesteps - 1], device=sample_batch['input_ids'].device)
                            noisy_ids, _ = self.diffuser(sample_batch['input_ids'][:1], t)
                            
                            # ノイズが追加されたトークンを表示
                            noisy_ids_list = noisy_ids[0].cpu().tolist()
                            print(f"ノイズ追加後のトークンID (最初の20個): {noisy_ids_list[:20]}")
                            
                            # マスクの数を確認
                            mask_count = noisy_ids_list.count(tokenizer.mask_token_id)
                            print(f"マスクトークンの数: {mask_count} / {len(noisy_ids_list)} ({mask_count/len(noisy_ids_list)*100:.1f}%)")
                            
                            # デコード結果
                            noisy_text = tokenizer.decode(noisy_ids_list)
                            print(f"ノイズ追加後テキスト: {noisy_text[:100]}..." if len(noisy_text) > 100 else noisy_text)
                            
                            # 明示的にマスクがあるか確認
                            if mask_count == 0:
                                print("警告: ノイズ追加後もマスクトークンが見つかりません！")
                                # 緊急対策としてノイズ関数を直接テスト (t=15)
                                print("-- 緊急ノイズチェック t=5 --")
                                high_t = 5  # 高いt値でテスト
                                manual_noisy = self.diffuser.add_noise(sample_batch['input_ids'][:1], high_t)
                                manual_mask_count = (manual_noisy == tokenizer.mask_token_id).sum().item()
                                print(f"手動ノイズ適用(t={high_t})後のマスク数: {manual_mask_count}")
                        else:
                            print("\nDiffusionノイズ追加テストは、学習開始後にサンプルとして実行されます")
                else:
                    print("トークナイザー情報が取得できません")
                
                print("ノイズ関数のテスト完了。学習を開始します。")
            except Exception as e:
                print(f"データ確認中にエラーが発生しました: {e}")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            # エポック表示を修正（1ベースではなくstart_epochから始まる）
            current_epoch = epoch + 1
            final_epoch = start_epoch + epochs
            
            # tqdmはAcceleratorと組み合わせて使用
            progress_bar = tqdm(
                dataloader, 
                desc=f"Diffusion Epoch {current_epoch}/{final_epoch}", 
                disable=not is_main_process
            )
            
            self.model.train()
            for batch_idx, batch in enumerate(progress_bar):
                # タイムステップをランダムに選択
                # 注意: t=0でもノイズが入るように修正しています
                t = torch.randint(0, diffuser.timesteps, (1,)).item()
                
                # 最初の100バッチは低ノイズ(t=0〜5)から始めて学習を安定させる
                if total_steps < 100:
                    t = min(5, t)  # 最初の100バッチは低ノイズから始める
                    if is_main_process and total_steps % 10 == 0:
                        print(f"[初期学習フェーズ] バッチ {total_steps}, ノイズレベル t={t}")
                
                # Acceleratorを使用した学習ステップ
                try:
                    with self.accelerator.accumulate(self.model):
                        step_loss = self._diffusion_train_step_accelerate(batch, diffuser, t, total_steps)
                        epoch_loss += step_loss
                    
                    # 最初の10ステップは学習状態を詳しくチェック
                    if is_main_process and total_steps < 10:
                        print(f"[デバッグ] ステップ {total_steps} 完了: 損失={step_loss:.4f}")
                        
                    # プログレスバーに現在の損失を表示
                    progress_bar.set_postfix(loss=f"{step_loss:.4f}")
                    
                    # 定期的なメモリ使用状況のログとチェックポイント保存
                    if is_main_process:
                        # メモリ使用状況をログ（10バッチごと）
                        if (total_steps + 1) % 10 == 0:
                            gpu_info = ""
                            try:
                                if torch.cuda.is_available():
                                    allocated = torch.cuda.memory_allocated() / 1024**3
                                    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                                    reserved = torch.cuda.memory_reserved() / 1024**3
                                    free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - allocated
                                    gpu_info = f", GPU: {allocated:.2f}GB/使用 {free:.2f}GB/空き (最大: {max_allocated:.2f}GB)"
                                    
                                    # メモリ消費が多い場合、警告を表示
                                    if allocated > 35.0:  # 40GBの87.5%
                                        print(f"警告: GPUメモリ使用量が高い ({allocated:.2f}GB/40GB)")
                            except Exception as e:
                                gpu_info = f", GPU情報取得エラー: {str(e)}"
                            
                            # 1イテレーションあたりの処理時間を推定
                            iter_time = 0
                            if hasattr(progress_bar, 'format_dict'):
                                iter_time = progress_bar.format_dict.get('elapsed', 0) / max(1, total_steps)
                            
                            print(f"ステップ {total_steps+1}, 損失: {step_loss:.4f}, 時間: {iter_time:.2f}秒/it{gpu_info}")
                        
                        # より頻繁にチェックポイントを保存（50バッチごと）
                        if (total_steps + 1) % 50 == 0:
                            try:
                                self.save_checkpoint(f"diffusion_step_{total_steps+1}")
                                print(f"ステップ {total_steps+1} でチェックポイントを保存しました")
                            except Exception as e:
                                print(f"チェックポイント保存エラー: {str(e)}")
                    
                except Exception as e:
                    if is_main_process:
                        print(f"ステップ {total_steps} でエラー発生: {str(e)}")
                        # 詳細なエラー情報
                        import traceback
                        traceback.print_exc()
                        
                        # 一時的にバッチを保存して後でデバッグできるようにする
                        try:
                            import pickle
                            debug_path = os.path.join(self.checkpoint_dir, f"debug_batch_{total_steps}.pkl")
                            with open(debug_path, 'wb') as f:
                                # CPU上に移動してからデータを保存
                                batch_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                                pickle.dump({'batch': batch_cpu, 't': t}, f)
                            print(f"デバッグ用にバッチを保存: {debug_path}")
                        except Exception as save_err:
                            print(f"デバッグデータ保存エラー: {save_err}")
                    
                    # 初期の数ステップでエラーが発生した場合は中断
                    if total_steps < 5:
                        raise e  # エラーを再度投げて中断
                    
                    # それ以外はエラーをスキップして続行
                    print("エラーをスキップして続行します")

                total_steps += 1

            # エポック統計（メインプロセスのみ）
            if is_main_process:
                avg_loss = epoch_loss / len(dataloader)
                epoch_time = time.time() - epoch_start_time
                print(f"Diffusion Epoch {current_epoch}/{final_epoch} done | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
                self.writer.add_scalar("Diffusion/Epoch Loss", avg_loss, current_epoch)
                self.writer.add_scalar("Diffusion/Epoch Time (s)", epoch_time, current_epoch)
            
            # スケジューラを進める
            self.scheduler.step()
            
            # チェックポイント保存（メインプロセスのみ）
            if is_main_process and ((current_epoch) % 5 == 0 or current_epoch == final_epoch):
                self.save_checkpoint(f"diffusion_epoch_{current_epoch}")
        
        # 学習完了（メインプロセスのみ）
        if is_main_process:
            total_time = time.time() - start_time
            self.writer.add_scalar("Diffusion/Total Training Time (min)", total_time / 60, 0)
            print(f"Diffusion training done in {total_time / 60:.2f} minutes")

    def _diffusion_train_step_accelerate(self, batch: Dict[str, Any], diffuser: SimpleTextDiffusion, t: int, step: int) -> float:
        """
        Acceleratorを使用したDiffusion学習ステップ
        TPU v5e-1向けに最適化
        """
        # input_idsとlabelsキーの存在確認 - Acceleratorは既にデバイスに配置済み
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
        elif "tokens" in batch:
            input_ids = batch["tokens"]
        else:
            raise ValueError("バッチにinput_idsまたはtokensキーが存在しません")
            
        # ラベルの処理
        if "labels" in batch:
            labels = batch["labels"]
        else:
            # ラベルが提供されていない場合は、diffuserを使って生成する
            # タイムステップをテンソルに変換してデバイスに配置
            t_tensor = torch.tensor([t], device=self.accelerator.device)
            noisy_tokens, labels = diffuser(input_ids.clone(), t_tensor)
        
        # オプティマイザのゼロ勾配（Acceleratorが管理）
        self.optimizer.zero_grad()
        
        # use_cut_cross_entropyフラグに基づいて処理を変更
        if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
            # Cut Cross Entropy用の処理
            # bfloat16の自動混合精度を使用
            with self.accelerator.autocast():
                embeddings = self.model(input_ids)
                classifier = self.model.get_classifier_weights()
                # TPUではbfloat16に変換
                if hasattr(torch, 'bfloat16'):
                    embeddings = embeddings.to(torch.bfloat16)
                    classifier = classifier.to(torch.bfloat16)
                else:
                    # bfloat16がない場合はfloat16を使用
                    embeddings = embeddings.half()
                    classifier = classifier.half()
                loss = linear_cross_entropy(embeddings, classifier, labels)
        else:
            # 通常のCross Entropy用の処理
            with self.accelerator.autocast():
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Acceleratorによる後方伝播と最適化
        self.accelerator.backward(loss)
        
        # 勾配クリッピング（設定されている場合）
        if hasattr(self.training_config, 'clip_value') and self.training_config.clip_value:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_config.clip_value)
        
        self.optimizer.step()
        
        # ロギング（メインプロセスのみ）
        if self.accelerator.is_main_process and step % 5 == 0:
            # 損失値を集約
            loss_value = self.accelerator.gather(loss).mean().item()
            self.writer.add_scalar("Diffusion/Loss", loss_value, step)
            self.writer.add_scalar("Diffusion/Timestep", t, step)
        
        # 損失値を返す前に集約（コミュニケーションコスト削減のため条件付き）
        if step % 5 == 0:
            return self.accelerator.gather(loss).mean().item()
        else:
            return loss.item()  # ローカルな損失値のみを返す

    def validate(self) -> float:
        """
        Acceleratorを使用したTPU v5e-1向け検証メソッド
        """
        if not self.valid_dataset:
            return 0.0
            
        self.model.eval()
        
        # マスターランク確認（メインプロセスのみ出力）
        is_main_process = self.accelerator.is_main_process
        
        # 検証用のバッチサイズは固定8に（TPU v5e-1向け）
        eval_batch_size = 8
        
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
        
        # DataLoaderを作成し、Acceleratorで準備
        dataloader = DataLoader(
            self.valid_dataset,
            batch_size=eval_batch_size, 
            shuffle=False,
            collate_fn=collator,
            num_workers=4,  # TPUではマルチプロセスデータロードが効果的
            pin_memory=True  # データ転送を高速化
        )
        dataloader = self.accelerator.prepare(dataloader)
        
        total_loss = 0.0
        total_batches = min(10, len(dataloader))  # 最大10バッチに制限
        
        # tqdmのプログレスバー
        from tqdm.auto import tqdm
        progress_bar = tqdm(
            dataloader, 
            desc="Validation", 
            total=total_batches, 
            disable=not is_main_process,
            leave=False
        )
        
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                # 最大10バッチまでで終了
                if i >= total_batches:
                    break
                
                # Acceleratorはbatchを既にデバイスに配置済み
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                
                try:
                    # use_cut_cross_entropyフラグに基づいて処理を変更
                    if hasattr(self.model, 'use_cut_cross_entropy') and self.model.use_cut_cross_entropy:
                        # Cut Cross Entropy用の処理 - TPU v5e-1向けにbfloat16を使用
                        with self.accelerator.autocast():
                            embeddings = self.model(input_ids)
                            classifier = self.model.get_classifier_weights()
                            # TPUではbfloat16に変換
                            if hasattr(torch, 'bfloat16'):
                                embeddings = embeddings.to(torch.bfloat16)
                                classifier = classifier.to(torch.bfloat16)
                            else:
                                # bfloat16がない場合はfloat16を使用
                                embeddings = embeddings.half()
                                classifier = classifier.half()
                            loss = linear_cross_entropy(embeddings, classifier, labels)
                    else:
                        # 通常のCross Entropy用の処理
                        with self.accelerator.autocast():
                            logits = self.model(input_ids)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
                    # 各プロセスで独立して損失を蓄積
                    total_loss += loss.item()
                    
                    # プログレスバーに現在の損失を表示
                    if is_main_process:
                        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                        
                except Exception as e:
                    if is_main_process:
                        print(f"WARNING: Error during validation: {str(e)}")
                    if i == 0:
                        # 最初のバッチでエラーが発生した場合、高い損失を返す
                        return 100.0
                    break
        
        # 処理したバッチ数で割る
        processed_batches = min(i + 1, total_batches)
        local_avg_loss = total_loss / processed_batches if processed_batches > 0 else 100.0
        
        # 全プロセスで平均損失を集約
        avg_loss_tensor = torch.tensor([local_avg_loss], device=self.accelerator.device)
        gathered_loss = self.accelerator.gather(avg_loss_tensor)
        avg_loss = gathered_loss.mean().item()
        
        # メインプロセスのみ出力
        if is_main_process:
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            print(f"Validation Loss: {avg_loss:.4f} [Perplexity: {perplexity:.2f}] (using {processed_batches} batches)")
        
        return avg_loss

    def save_checkpoint(self, name: str) -> None:
        """
        Acceleratorを使用したチェックポイント保存（TPU v5e-1対応）
        メインプロセスのみがチェックポイントを保存
        """
        # メインプロセスのみが保存する
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
            
            # 通常のチェックポイント情報
            checkpoint = {
                "model_config": self.model.config,
                "training_config": self.training_config,
            }
            
            # Accelerator経由でモデルを抽出して保存
            # unwrap_model()でDistributedDataParallelなどからモデルを取り出す
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            checkpoint["model_state_dict"] = unwrapped_model.state_dict()
            
            # オプティマイザは保存しないことも可能（TPU環境では省略することもある）
            # 保存する場合はAcceleratorから取り出して保存
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            
            # 保存
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
        # 全プロセスが同期するのを待つ
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: str) -> int:
        """
        Acceleratorを使用したチェックポイント読み込み（TPU v5e-1対応）
        チェックポイントの次のエポック番号を返す
        """
        # 全プロセスがロードする必要がある
        checkpoint = torch.load(path, map_location=self.accelerator.device)
        
        # まずAcceleratorからオリジナルモデルを取得
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # 状態辞書をロード
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        
        # オプティマイザもロードする場合
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # チェックポイントのファイル名からエポック番号を抽出
        import re
        start_epoch = 0
        epoch_match = re.search(r'diffusion_epoch_(\d+)', path)
        if epoch_match:
            start_epoch = int(epoch_match.group(1))
            if self.accelerator.is_main_process:
                print(f"チェックポイントのエポック番号: {start_epoch}、次のエポック: {start_epoch + 1}")
        
        # メインプロセスのみが出力
        if self.accelerator.is_main_process:
            print(f"Model loaded from {path}")
        
        # 全プロセスが同期するのを待つ
        self.accelerator.wait_for_everyone()
        
        # 抽出したエポック番号を返す
        return start_epoch

    def close(self) -> None:
        """
        リソースを解放
        """
        if self.accelerator.is_main_process:
            self.writer.close()
            print("TensorBoard writer closed and resources released")
            
        # 全プロセスが同期するのを待つ
        self.accelerator.wait_for_everyone()