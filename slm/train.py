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

    def train_diffusion(self, num_epochs: Optional[int] = None, start_epoch: int = 0, start_step: int = 0) -> None:
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
        
        total_steps = start_step
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
                tokenizer = self.model.config.tokenizerもしもし if hasattr(self.model.config, 'tokenizer') else None
                
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
                
                try:
                    # Accelerator準備済みの関数を使用して学習ステップを実行
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
                    # TPUif total_steps < 5:
                    #     raise e  # エラーを再度投げて中断
                    
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
                # エポック情報を含むチェックポイント名
                checkpoint_name = f"model_e{current_epoch:03d}"
                
                # 検証データがあれば評価する
                try:
                    if self.valid_dataset is not None:
                        print(f"エポック {current_epoch} の検証中...")
                        val_loss = self.validate()
                        print(f"検証損失: {val_loss:.4f}")
                        self.writer.add_scalar("Diffusion/Validation Loss", val_loss, current_epoch)
                        
                        # 検証損失を含むチェックポイント名
                        checkpoint_name = f"model_e{current_epoch:03d}_step{total_steps:04d}_loss{val_loss:.4f}"
                except Exception as e:
                    print(f"検証中にエラーが発生しました: {e}")
                
                # チェックポイント保存
                self.save_checkpoint(checkpoint_name)
                
                # if current_epoch % 10 == 0 or current_epoch == final_epoch:
                print(f"エポック {current_epoch} のチェックポイントをfinal_model.ptとしても保存")
                self.save_checkpoint("final_model")
        
        # 学習完了（メインプロセスのみ）
        if is_main_process:
            total_time = time.time() - start_time
            self.writer.add_scalar("Diffusion/Total Training Time (min)", total_time / 60, 0)
            print(f"Diffusion training done in {total_time / 60:.2f} minutes")

    def _diffusion_train_step_accelerate(self, batch: Dict[str, Any], diffuser: SimpleTextDiffusion, t: int, step: int) -> float:
        """
        Acceleratorを使用したDiffusion学習ステップ
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
        
        # GPU/CPU互換性のあるモードでの処理
        try:
            # Cut Cross Entropyを使用（必須）
            with self.accelerator.autocast():
                embeddings = self.model(input_ids)
                classifier = self.model.get_classifier_weights()
                
                # cudaの場合はfloat16に変換
                if input_ids.is_cuda:
                    embeddings = embeddings.half()
                    classifier = classifier.half()
                
                # tritonドライバーエラーを避けるためにtry-exceptで囲む
                try:
                    # cut_cross_entropyモジュールを使用
                    loss = linear_cross_entropy(embeddings, classifier, labels)
                except RuntimeError as e:
                    if "active drivers" in str(e):
                        # Tritonエラーの場合、一時的なフォールバック
                        print("WARNING: Tritonドライバーエラー発生、一時的なフォールバック処理を実行")
                        logits = torch.matmul(embeddings, classifier.t())
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                    else:
                        raise e
        except Exception as e:
            print(f"Cut Cross Entropyでエラー発生: {e}")
            # エラーを上位に伝播させる
            raise e
        
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
        Acceleratorを使用した検証メソッド
        
        Returns:
            float: 検証データセットに対する平均損失。エラー発生時は100.0を返す。
        """
        if not self.valid_dataset:
            return 0.0
            
        self.model.eval()
        
        # マスターランク確認（メインプロセスのみ出力）
        is_main_process = self.accelerator.is_main_process
        
        # 検証用のバッチサイズ
        eval_batch_size = 8
        
        try:
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
                num_workers=4,
                pin_memory=True
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
            
            if is_main_process:
                print(f"バリデーション開始 (最大{total_batches}バッチ)")
            
            with torch.no_grad():
                for i, batch in enumerate(progress_bar):
                    # 最大10バッチまでで終了
                    if i >= total_batches:
                        break
                    
                    try:
                        # バッチ内のテンソルサイズを安全にチェック
                        if not all(key in batch for key in ["input_ids", "labels"]):
                            if is_main_process:
                                print(f"バッチ {i} に必要なキー(input_ids, labels)が含まれていません")
                            continue
                            
                        input_ids = batch["input_ids"]
                        labels = batch["labels"]
                        
                        # 不正な値やNaNをチェック
                        if torch.isnan(input_ids).any() or torch.isnan(labels).any():
                            if is_main_process:
                                print(f"バッチ {i} にNaN値が含まれています - スキップします")
                            continue
                            
                        # 範囲外のインデックスチェック
                        vocab_size = self.model.get_classifier_weights().size(0)
                        if labels.max() >= vocab_size:
                            if is_main_process:
                                print(f"バッチ {i} に語彙サイズ外のインデックス値があります: max={labels.max().item()}, vocab_size={vocab_size}")
                                # 不正なインデックスをマスク処理
                                labels = torch.where(labels < vocab_size, labels, -100)
                        
                        # Cut Cross Entropyを優先的に使用
                        with self.accelerator.autocast():
                            embeddings = self.model(input_ids)
                            classifier = self.model.get_classifier_weights()
                            
                            # デバイスによって適切な精度に変換
                            if input_ids.device.type == "cuda":
                                embeddings = embeddings.half()  # CUDA用にfloat16
                                classifier = classifier.half()
                            elif hasattr(torch, 'bfloat16') and self.accelerator.device.type == "xla":
                                embeddings = embeddings.to(torch.bfloat16)  # TPU用にbfloat16
                                classifier = classifier.to(torch.bfloat16)
                            
                            # tritonドライバーエラーを避けるためにtry-exceptで囲む
                            try:
                                # cut_cross_entropyモジュールを使用
                                from cut_cross_entropy import linear_cross_entropy
                                loss = linear_cross_entropy(embeddings, classifier, labels)
                            except RuntimeError as e:
                                if "active drivers" in str(e) or "device-side assert" in str(e):
                                    # Tritonエラーの場合、一時的なフォールバック
                                    if is_main_process:
                                        print("WARNING: 特殊エラー発生、標準のcross_entropyにフォールバック")
                                    # 標準の実装にフォールバック
                                    logits = torch.matmul(embeddings, classifier.t())
                                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                                else:
                                    raise e
                        
                        # 各プロセスで独立して損失を蓄積
                        total_loss += loss.item()
                        
                        # プログレスバーに現在の損失を表示
                        if is_main_process:
                            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                            
                    except Exception as e:
                        if is_main_process:
                            import traceback
                            print(f"バリデーションのバッチ {i} でエラー発生: {str(e)}")
                            traceback.print_exc()
                        if i == 0:
                            # 最初のバッチでエラーが発生した場合、高い損失を返す
                            return 100.0
                        # 他のバッチは続行
                        continue
                
                # 処理したバッチ数で割る
                processed_batches = min(i + 1, total_batches)
                if processed_batches <= 0:  # すべてのバッチでエラーが発生した場合
                    return 100.0
                    
                local_avg_loss = total_loss / processed_batches
            
            # 全プロセスで平均損失を集約
            try:
                avg_loss_tensor = torch.tensor([local_avg_loss], device=self.accelerator.device)
                gathered_loss = self.accelerator.gather(avg_loss_tensor)
                avg_loss = gathered_loss.mean().item()
            except Exception as e:
                if is_main_process:
                    print(f"損失の集約でエラー発生: {e}")
                avg_loss = local_avg_loss  # フォールバック: ローカルの損失を使用
            
            # メインプロセスのみ出力
            if is_main_process:
                try:
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    print(f"Validation Loss: {avg_loss:.4f} [Perplexity: {perplexity:.2f}] (using {processed_batches} batches)")
                except Exception as e:
                    print(f"パープレキシティ計算でエラー: {e}")
            
            return avg_loss
            
        except Exception as e:
            # validate全体でエラーが発生した場合
            if is_main_process:
                import traceback
                print(f"バリデーション実行中に重大なエラー発生: {str(e)}")
                traceback.print_exc()
            return 100.0

    def save_checkpoint(self, name: str, current_epoch: int = None, current_step: int = None, val_loss: float = None) -> None:
        """
        Acceleratorを使用したチェックポイント保存（TPU v5e-1対応）
        メインプロセスのみがチェックポイントを保存
        
        Args:
            name: チェックポイントの名前 (拡張子なし)
            current_epoch: 現在のエポック番号 (オプション)
            current_step: 現在のステップ数 (オプション)
            val_loss: 検証損失値 (オプション)
        """
        # メインプロセスのみが保存する
        if self.accelerator.is_main_process:
            try:
                # チェックポイントパス
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
                
                # 学習情報も含めたチェックポイント辞書を作成
                checkpoint = {
                    "model_config": self.model.config,
                    "training_config": self.training_config,
                    "epoch": current_epoch,
                    "step": current_step,
                    "val_loss": val_loss,
                    "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                # Accelerator経由でモデルを抽出して保存
                # unwrap_model()でDistributedDataParallelなどからモデルを取り出す
                try:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    
                    # まずCPUに移動する処理を安全に行う
                    try:
                        model_state_dict = {
                            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in unwrapped_model.state_dict().items()
                        }
                        checkpoint["model_state_dict"] = model_state_dict
                    except RuntimeError as e:
                        if "device-side assert" in str(e) or "CUDA" in str(e):
                            print(f"CUDAエラーが発生しましたが、パラメータを一つずつCPUに移動して再試行します: {e}")
                            # テンソルごとに個別に移動を試みる
                            model_state_dict = {}
                            for k, v in unwrapped_model.state_dict().items():
                                try:
                                    if isinstance(v, torch.Tensor):
                                        model_state_dict[k] = v.detach().cpu()
                                    else:
                                        model_state_dict[k] = v
                                except Exception:
                                    # 個別のテンソルで失敗した場合、そのテンソルはスキップ
                                    print(f"警告: パラメータ {k} のCPU移動に失敗しました。スキップします。")
                            checkpoint["model_state_dict"] = model_state_dict
                        else:
                            raise e
                            
                except Exception as e:
                    print(f"モデル状態の抽出中にエラー: {e}")
                    # 最小限のチェックポイントとして続行
                    checkpoint["model_state_dict_error"] = str(e)
                
                # オプティマイザの状態を安全に保存
                try:
                    optimizer_state = self.optimizer.state_dict()
                    # パラメーターの'state'部分だけCPUに移動
                    if 'state' in optimizer_state:
                        cpu_state = {}
                        for param_id, param_state in optimizer_state['state'].items():
                            cpu_param_state = {}
                            for k, v in param_state.items():
                                if isinstance(v, torch.Tensor):
                                    try:
                                        cpu_param_state[k] = v.detach().cpu()
                                    except Exception:
                                        print(f"警告: オプティマイザのテンソル {k} のCPU移動に失敗しました。スキップします。")
                                else:
                                    cpu_param_state[k] = v
                            cpu_state[param_id] = cpu_param_state
                        optimizer_state['state'] = cpu_state
                    checkpoint["optimizer_state_dict"] = optimizer_state
                except Exception as e:
                    print(f"オプティマイザ状態の抽出中にエラー: {e}")
                    # オプティマイザ状態を保存せずに続行
                    checkpoint["optimizer_state_dict_error"] = str(e)
                
                # 保存とバックアップ
                try:
                    # まずバックアップファイル名を作成
                    backup_path = os.path.join(self.checkpoint_dir, f"{name}_backup.pt")
                    
                    # チェックポイントを保存
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                    
                except Exception as e:
                    print(f"チェックポイント保存中にエラー: {e}")
                    
                    # メタデータだけでも保存を試みる
                    try:
                        meta_checkpoint = {
                            "model_config": self.model.config,
                            "training_config": self.training_config,
                            "epoch": current_epoch,
                            "step": current_step,
                            "val_loss": val_loss,
                            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "save_error": str(e)
                        }
                        meta_path = os.path.join(self.checkpoint_dir, f"{name}_meta_only.pt")
                        torch.save(meta_checkpoint, meta_path)
                        print(f"メタデータのみのチェックポイントを保存: {meta_path}")
                    except Exception as e2:
                        print(f"メタデータ保存中にもエラー: {e2}")
                
            except Exception as e:
                import traceback
                print(f"チェックポイント保存プロセス全体でエラー: {e}")
                traceback.print_exc()
        
        # 全プロセスが同期するのを待つ
        try:
            self.accelerator.wait_for_everyone()
        except Exception as e:
            print(f"プロセス同期中にエラー: {e}")
            # 同期エラーは無視して続行

    def load_checkpoint(self, path: str) -> tuple[int, int]:
        """
        Acceleratorを使用したチェックポイント読み込み（TPU v5e-1対応）
        チェックポイントの次のエポック番号とステップ数を返す
        
        Args:
            path: チェックポイントファイルのパス
            
        Returns:
            tuple[int, int]: (start_epoch, start_step) - 次のエポック番号とステップ数
        
        PyTorch 2.6+対応:
        - weights_only=Falseを明示的に指定してクラス情報も含めて読み込む
        - または、add_safe_globalsを使用して許可リストに追加
        """
        # PyTorch 2.6+対応: 必要なクラスを安全なグローバルとして登録
        from slm.config import ModelConfig, TrainingConfig
        import torch.serialization
        
        # T5Tokenizerも明示的に追加
        try:
            from transformers.models.t5.tokenization_t5 import T5Tokenizer
            torch.serialization.add_safe_globals([ModelConfig, TrainingConfig, T5Tokenizer])
        except ImportError:
            # transformersライブラリがインストールされていない場合はスキップ
            torch.serialization.add_safe_globals([ModelConfig, TrainingConfig])
        
        # デフォルト値を事前に設定しておく（チェックポイントから取得できない場合用）
        start_epoch = 0
        start_step = 0
        
        try:
            # まず安全なグローバルリストを使用して読み込み
            checkpoint = torch.load(path, map_location=self.accelerator.device)
        except Exception as e:
            # 失敗した場合、weights_only=Falseで再試行（信頼できるチェックポイントの場合）
            if self.accelerator.is_main_process:
                print(f"通常の読み込みに失敗しました。weights_only=Falseで再試行します: {e}")
            try:
                checkpoint = torch.load(path, map_location=self.accelerator.device, weights_only=False)
            except Exception as e2:
                if self.accelerator.is_main_process:
                    print(f"weights_only=Falseでも失敗しました。チェックポイントを読み込めません: {e2}")
                    return start_epoch, start_step  # デフォルト値を返す
        
        # まずAcceleratorからオリジナルモデルを取得
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # 状態辞書をロード
        try:
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"モデル状態のロードに失敗しました: {e}")
        
        # オプティマイザもロードする場合
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"オプティマイザ状態のロードに失敗しました。学習率のみ復元します: {e}")
                    # オプティマイザの学習率だけ抽出して設定
                    for param_group in self.optimizer.param_groups:
                        if "lr" in checkpoint.get("optimizer_state_dict", {}).get("param_groups", [{}])[0]:
                            param_group['lr'] = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
        
        # チェックポイントのファイル名やメタデータからエポック番号とステップ数を抽出
        import re
        
        # チェックポイントから直接情報を取得（Noneチェック付き）
        if "epoch" in checkpoint and checkpoint["epoch"] is not None:
            start_epoch = checkpoint["epoch"]
        
        if "step" in checkpoint and checkpoint["step"] is not None:
            start_step = checkpoint["step"]
        
        # ファイル名からのエポック/ステップ情報抽出（メタデータがない場合のバックアップ）
        if start_epoch == 0:
            # diffusion_epoch_X パターンをチェック
            epoch_match = re.search(r'diffusion_epoch_(\d+)', path)
            if epoch_match:
                start_epoch = int(epoch_match.group(1))
        
        if start_step == 0:
            # diffusion_step_X パターンをチェック
            step_match = re.search(r'diffusion_step_(\d+)', path)
            if step_match:
                start_step = int(step_match.group(1))
                
                # エポックが取得できていない場合は、ステップからエポックを推定
                if start_epoch == 0:
                    # おおよそのバッチサイズとデータセットサイズからエポックを推定
                    batch_size = getattr(self.training_config, 'batch_size', 8)
                    dataset_size = len(self.train_dataset) if self.train_dataset else 1000
                    approx_steps_per_epoch = max(1, dataset_size // batch_size)
                    start_epoch = start_step // approx_steps_per_epoch
                    if self.accelerator.is_main_process:
                        print(f"ステップ {start_step} からエポックを推定: {start_epoch}")
        
        # メインプロセスのみが出力
        if self.accelerator.is_main_process:
            print(f"チェックポイント {path} を読み込みました")
            print(f"次のエポック: {start_epoch + 1}, 開始ステップ: {start_step}")
        
        # 全プロセスが同期するのを待つ
        self.accelerator.wait_for_everyone()
        
        # エポック番号とステップ数のタプルを返す
        return start_epoch, start_step

    def close(self) -> None:
        """
        リソースを解放
        """
        if self.accelerator.is_main_process:
            self.writer.close()
            print("TensorBoard writer closed and resources released")
            
        # 全プロセスが同期するのを待つ
        self.accelerator.wait_for_everyone()