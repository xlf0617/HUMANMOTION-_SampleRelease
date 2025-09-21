import json
import math
import os
import random
from pathlib import Path

from torch.utils.data.distributed import DistributedSampler
from accelerate.data_loader import DataLoaderShard

from config import TrainingConfig
# import bitsandbytes as bnb
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers import DDPMScheduler, EulerDiscreteScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPVisionModelWithProjection, WhisperModel, WhisperProcessor, AutoFeatureExtractor
from einops import rearrange, repeat
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import argparse
import logging
from model.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, add_ip_adapters
from model.audio_adapter.audio_proj import AudioProjModel
from model.audio_adapter.audio_to_bucket import Audio2bucketModel
from model.face_align.align import AlignImage
# import wandb
from model.controlnet_for_sonic import ControlNetSonicModel

logger = get_logger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class SonicTrainingDataset(Dataset):
    """优化后的数据集，使用缓存的音频特征，支持多个数据集文件夹"""

    def __init__(self, data_roots, image_size=512, num_train_frames=14):
        """
        初始化数据集
        
        Args:
            data_roots: 数据集根路径列表，可以是字符串列表或Path对象列表
            image_size: 图像大小
            num_train_frames: 训练帧数
        """
        # 确保data_roots是列表形式
        if isinstance(data_roots, (str, Path)):
            data_roots = [data_roots]

        # 转换为Path对象列表
        self.data_roots = [Path(root) for root in data_roots]
        self.image_size = image_size
        self.num_train_frames = num_train_frames

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.samples = self._scan_samples()
        logger.info("Loaded %d training samples from %d datasets", 
                   len(self.samples), len(self.data_roots))

    def _scan_samples(self):
        """扫描所有数据集文件夹中的样本"""
        samples = []

        for data_root in self.data_roots:
            if not data_root.exists():
                logger.warning("Dataset directory %s does not exist, skipping", data_root)
                continue

            logger.info("Scanning dataset directory: %s", data_root)
            dataset_samples = 0

            for sample_dir in data_root.glob("*/"):
                if not sample_dir.is_dir():
                    continue

                if (sample_dir / "metadata.json").exists():
                    try:
                        with open(sample_dir / "metadata.json") as f:
                            metadata = json.load(f)

                        required = ["reference.jpg", "audio.wav", "mask.png", "frames", "skeletons", "audio_features.pt"]
                        if all((sample_dir / f).exists() for f in required):
                            samples.append({
                                "dir": sample_dir, 
                                "meta": metadata,
                                "dataset_root": data_root  # 记录所属的数据集根目录
                            })
                            dataset_samples += 1
                    except Exception as e:
                        logger.warning("Failed to process sample %s: %s", sample_dir, e)

            logger.info("Found %d samples in %s", dataset_samples, data_root)

        return samples

    def __len__(self):
        return len(self.samples)

    def img_crop(self, img, target_size):
        w, h = img.size
        half_size = target_size // 2
        if w > target_size:
            x_center = random.randint(half_size, w - half_size)
            x = x_center - half_size
            X = x_center + half_size
        else:
            x = 0
            X = w
        if h > target_size:
            y_center = random.randint(half_size, h - half_size)
            y = y_center - half_size
            Y = y_center + half_size
        else:
            y = 0
            Y = h
        return x, y, X, Y

    # noinspection DuplicatedCode
    def __getitem__(self, idx):
        sample = self.samples[idx]
        d = sample["dir"]
        meta = sample["meta"]

        try:
            # 参考图像处理
            ref_img = Image.open(d / "reference.jpg").convert('RGB')
            w, h = ref_img.size
            scale = self.image_size / min(w, h)
            new_w = round(w * scale / 64) * 64
            new_h = round(h * scale / 64) * 64

            # new_w = min(512, new_w)
            # new_h = min(512, new_h)
            ref_img = ref_img.resize((new_w, new_h), Image.LANCZOS)

            need_crop = False
            if new_w > self.image_size or new_h > self.image_size:
                need_crop = True
                x, y, X, Y = self.img_crop(ref_img, self.image_size)
            else:
                need_crop = False

            ref_img = ref_img.crop((x, y, X, Y)) if need_crop else ref_img
            ref_img = self.image_transform(ref_img)

            # CLIP图像处理
            clip_img = Image.open(d / "reference.jpg").convert('RGB')
            clip_img = clip_img.resize((new_w, new_h), Image.LANCZOS)
            clip_img = clip_img.crop((x, y, X, Y)) if need_crop else clip_img
            clip_img = clip_img.resize((224, 224), Image.LANCZOS)
            clip_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(clip_img)

            # 面部遮罩
            mask = Image.open(d / "mask.png").convert("L")
            mask = mask.resize((new_w, new_h), Image.LANCZOS)
            mask = mask.crop((x, y, X, Y)) if need_crop else mask
            mask = self.mask_transform(mask)

            # 视频帧
            frame_paths = sorted((d / "frames").glob("*.jpg"))

            # 骨骼图序列
            skeleton_paths = sorted((d / "skeletons").glob("*.jpg"))

            # 加载缓存的音频特征
            audio_cache = torch.load(d / "audio_features.pt", weights_only=True)
            audio_f = audio_cache["features"]

            # 对齐特征，确保无偏移
            mapping = 1

            max_valid_start = len(frame_paths) - self.num_train_frames
            # 确保 max_valid_start 也能被 mapping 整除，否则向下取整
            max_valid_start = max_valid_start // mapping * mapping
            start_id = np.random.randint(0, max_valid_start // mapping + 1) * mapping

            # num_train_frames对应的音频特征长度
            num_train_audio = 100

            frames = []
            skeletons = []
            count_frame = 0
            for idx, p in enumerate(frame_paths):
                if idx >= start_id:
                    frame = Image.open(p).convert('RGB')
                    frame = frame.resize((new_w, new_h), Image.LANCZOS)
                    frame = frame.crop((x, y, X, Y)) if need_crop else frame
                    frame = self.image_transform(frame)
                    frames.append(frame)
                    count_frame += 1
                    if count_frame >= self.num_train_frames:
                        break
            frames = torch.stack(frames)

            count_skeleton = 0
            for idx, p in enumerate(skeleton_paths):
                if idx >= start_id:
                    skeleton = Image.open(p).convert('RGB')
                    skeleton = skeleton.resize((new_w, new_h), Image.LANCZOS)
                    skeleton = skeleton.crop((x, y, X, Y)) if need_crop else skeleton
                    skeleton = self.image_transform(skeleton)
                    skeletons.append(skeleton)
                    count_skeleton += 1
                    if count_skeleton >= self.num_train_frames:
                        break
            skeletons = torch.stack(skeletons)
            # print('----------------------------------------------------')
            # print(frame_paths)
            # print(skeleton_paths)
            # # assert 0
            # # 保存调试图片 - 检查数据匹配
            # import os
            # debug_dir = "/root/autodl-tmp/Sonic_Train_max/debug_images"
            # os.makedirs(debug_dir, exist_ok=True)

            # # 保存原始视频帧（UNet输入）
            # for i, frame in enumerate(frames):
            #     # 反归一化并保存
            #     frame_np = frame.permute(1, 2, 0).numpy()
            #     frame_np = (frame_np * 255).astype('uint8')
            #     frame_img = Image.fromarray(frame_np)
            #     frame_img.save(f"{debug_dir}/unet_input_frame_{i}.png")

            # # 保存骨架图（ControlNet输入）
            # for i, skeleton in enumerate(skeletons):
            #     # 反归一化并保存
            #     skeleton_np = skeleton.permute(1, 2, 0).numpy()
            #     skeleton_np = (skeleton_np * 255).astype('uint8')
            #     skeleton_img = Image.fromarray(skeleton_np)
            #     skeleton_img.save(f"{debug_dir}/controlnet_input_skeleton_{i}.png")
            # print(f"已保存调试图片到 {debug_dir}")
            # print(f"UNet输入帧数: {len(frames)}, ControlNet输入骨架数: {len(skeletons)}")
            # print(f"帧形状: {frames.shape}, 骨架形状: {skeletons.shape}")
            # assert 0
            audio_start = int(audio_f.shape[-1] * start_id / len(frame_paths))
            audio_end = audio_start + num_train_audio
            audio_feature = audio_f[:, :, audio_start:audio_end]

            return {
                "ref_image": ref_img,
                "clip_images": clip_img,
                "face_mask": mask,
                "video_frames": frames,
                "skeleton_frames": skeletons,
                "audio_features": audio_feature,
                "width": self.image_size,
                "height": self.image_size,
                "dataset_root": str(sample["dataset_root"]),  # 添加数据集来源信息
                "sample_path": str(d),
            }
        except Exception as e:
            # assert 0
            logger.error("Failed to load sample %s: %s", d, e)
            return self[(idx + 1) % len(self)]

# noinspection DuplicatedCode,PyTypeChecker
class SonicTrainer:
    """Sonic训练器"""

    # noinspection PyTypeChecker
    def __init__(self, config, device=None):
        self.config = config

        # # 显存优化策略选择
        # if hasattr(config, 'memory_optimization') and config.memory_optimization == "balanced":
        #     # 平衡优化：禁用混合精度以避免GradScaler问题
        #     config.mixed_precision = "no"
        #     config.weight_dtype = "fp32"
        #     print("⚠️ 平衡显存优化模式：禁用混合精度以避免GradScaler问题")
        # elif hasattr(config, 'memory_optimization') and config.memory_optimization == "fp16_safe":
        #     # FP16安全模式：使用FP16但添加安全措施
        #     config.mixed_precision = "fp16"
        #     config.weight_dtype = "fp16"
        #     config.learning_rate = 1e-5  # 进一步降低学习率
        #     config.max_grad_norm = 10.0  # 提高梯度裁剪阈值
        #     print("⚠️ FP16安全模式：使用FP16，降低学习率，提高梯度裁剪阈值")
        # else:
        #     # 默认：禁用混合精度
        #     config.mixed_precision = "no"
        #     config.weight_dtype = "fp32"
        #     print("✓ 默认模式：禁用混合精度训练以避免FP16梯度问题")

        # 初始化加速器 - 使用多GPU配置
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None,
        )

        # 设置设备
        self.device = self.accelerator.device

        # 设置日志
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        if self.accelerator.is_main_process:
            logger.info(f"Using {self.accelerator.num_processes} GPUs for training")

        # 初始化模型
        self._init_models(config)

        # 初始化数据加载器
        # self._init_dataloaders()

        # 初始化优化器和调度器
        self._init_optimizer_scheduler()

        # 准备训练
        self._prepare_training()

    def _init_models(self, config):
        """初始化模型"""
        logger.info("Loading models...")

        # 加载VAE
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.config.model_path,
            subfolder="vae",
            variant="fp16"
        )
        self.vae.requires_grad_(False)

        # 加载CLIP
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            self.config.model_path,
            subfolder="image_encoder",
            variant="fp16"
        )
        self.clip_model.requires_grad_(False)

        # 噪声调度器
        self.train_scheduler = DDPMScheduler.from_pretrained("./scheduler")

        logger.info(self.train_scheduler.config)  # 打印整个配置字典

        # 加载UNet
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            self.config.model_path,
            subfolder="unet",
        )

        add_ip_adapters(self.unet, [32], [config.ip_audio_scale])

        self.audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024,
                                          context_tokens=32)
        self.audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024,
                                              intermediate_dim=1024,
                                              output_dim=1, context_tokens=2)

        # 加载预训练权重
        if self.config.unet_checkpoint_path:
            self.unet.load_state_dict(
                # torch.load(self.config.unet_checkpoint_path, map_location="cpu", weights_only=True),
                torch.load(
                    "/root/autodl-tmp/Sonic_Train_max/resume_checkpoints2/3epoch_final_model/unet/pytorch_model.bin",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )

        if self.config.audio2token_checkpoint_path:
            self.audio2token.load_state_dict(
                torch.load(self.config.audio2token_checkpoint_path, map_location="cpu", weights_only=True),
                strict=True,
            )

        if self.config.audio2bucket_checkpoint_path:
            self.audio2bucket.load_state_dict(
                torch.load(self.config.audio2bucket_checkpoint_path, map_location="cpu", weights_only=True),
                strict=True,
            )

        # if self.config.controlnet_model_name_or_path:
        if True:
            logger.info("Loading existing controlnet weights")
            # self.controlnet = ControlNetSonicModel.from_pretrained(self.config.controlnet_model_name_or_path)
            self.controlnet = ControlNetSonicModel.from_pretrained(
                "/root/autodl-tmp/Sonic_Train_max/resume_checkpoints2/3epoch_final_model/controlnet"
            )
        else:
            logger.info("Initializing controlnet weights from unet")
            self.controlnet = ControlNetSonicModel.from_unet(self.unet)

        self.audio2token.requires_grad_(False)
        self.audio2bucket.requires_grad_(False)

        if config.weight_dtype == "fp16":
            self.weight_dtype = torch.float16
        elif config.weight_dtype == "fp32":
            self.weight_dtype = torch.float32
        elif config.weight_dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Do not support weight dtype: {config.weight_dtype} during training"
            )

        # 冻结原始权重，只训练IP-Adapter
        for n, p in self.unet.named_parameters():
            if "to_k_ip" in n or "to_v_ip" in n:
                p.requires_grad_(True)
                # p.requires_grad_(False)  # 禁用IP-Adapter训练
            else:
                p.requires_grad_(False)

        # 设置ControlNet参数为可训练
        self.controlnet.requires_grad_(True)

        self.whisper_model = WhisperModel.from_pretrained(self.config.whisper_path).eval()
        self.whisper_model.requires_grad_(False)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.whisper_path)

        # 移动模型到对应设备和精度
        self.vae = self.vae.to(self.weight_dtype)
        self.clip_model = self.clip_model.to(self.weight_dtype)

        # 启用梯度检查点以节省显存
        try:
            # 方法1：新版本API
            self.unet.enable_gradient_checkpointing()
        except TypeError:
            try:
                # 方法2：旧版本API
                self.unet._set_gradient_checkpointing(True)
            except Exception as e:
                # 方法3：直接设置属性
                if hasattr(self.unet, 'gradient_checkpointing'):
                    self.unet.gradient_checkpointing = True
                else:
                    logger.warning(f"无法启用梯度检查点: {e}")

        # 同样处理ControlNet
        try:
            if hasattr(self.controlnet, 'enable_gradient_checkpointing'):
                self.controlnet.enable_gradient_checkpointing()
            elif hasattr(self.controlnet, '_set_gradient_checkpointing'):
                self.controlnet._set_gradient_checkpointing(True)
            elif hasattr(self.controlnet, 'gradient_checkpointing'):
                self.controlnet.gradient_checkpointing = True
        except Exception as e:
            logger.warning(f"无法启用ControlNet梯度检查点: {e}")

    def _init_dataloaders(self):
        """初始化数据加载器"""
        # 训练数据集
        train_dataset = SonicTrainingDataset(
            data_roots=self.config.data_root,
            num_train_frames=self.config.num_train_frames,
        )

        # 创建基础DataLoader
        base_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,  # 重要：在这里设置为False
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        return base_dataloader

    def _init_optimizer_scheduler(self):
        """初始化优化器和学习率调度器"""
        # 收集所有需要优化的参数
        params_to_optimize = [
            {"params": [p for p in self.unet.parameters() if p.requires_grad], "lr": self.config.ip_learning_rate},
            {"params": [p for p in self.controlnet.parameters() if p.requires_grad], "lr": self.config.controlnet_learning_rate},
        ]

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            # lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

    def _prepare_training(self):
        base_dataloader = self._init_dataloaders()

        """准备训练 - 使用accelerator准备所有组件"""
        # 准备模型、优化器、数据加载器、学习率调度器
        (self.unet, self.controlnet, self.optimizer, self.train_dataloader) = (
            self.accelerator.prepare(
                self.unet, self.controlnet, self.optimizer, base_dataloader
            )
        )

        # 准备非训练模型
        (self.vae, self.clip_model, self.whisper_model, 
         self.audio2token, self.audio2bucket) = self.accelerator.prepare(
            self.vae, self.clip_model, self.whisper_model,
            self.audio2token, self.audio2bucket
        )

        # 计算总训练步数
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        self.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_scheduler(
            # self.config.lr_scheduler,
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.max_train_steps,
        )
        # 4. 把 scheduler 也 prepare 一下（accelerator 会处理 scaler 等）
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

        if self.accelerator.is_main_process:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
            logger.info(f"  Num examples per GPU = {len(self.train_dataloader)}")
            logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = "
                        f"{self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps}")
            logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {self.max_train_steps}")

    def train(self):
        """主训练循环"""
        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(
            range(global_step, self.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.config.num_train_epochs):
            self.unet.train()
            self.controlnet.train()
            train_loss = 0.0

            # # 或者如果使用accelerator的prepare_data_loader
            # if hasattr(self.train_dataloader, "sampler"):
            #     self.train_dataloader.sampler.set_epoch(epoch)
            if isinstance(self.train_dataloader, DataLoaderShard):
                logger.info(f"[Rank {self.accelerator.process_index}] Setting epoch for DataLoaderShard")
                self.train_dataloader.set_epoch(epoch)

            for step, batch in enumerate(self.train_dataloader):
                if step == 0 and epoch == 0:
                    print(f"[Rank {self.accelerator.process_index}] First sample: {batch['sample_path'][0]}", flush=True)

                if step % 5 == 0:
                    torch.cuda.empty_cache()

                with self.accelerator.accumulate(self.unet, self.controlnet):
                    # with self.accelerator.accumulate(self.controlnet):
                    # 前向传播
                    with self.accelerator.autocast():
                        # 获取数据
                        ref_image = batch['ref_image']
                        clip_image = batch['clip_images']
                        face_mask = batch['face_mask']
                        video = batch['video_frames']
                        audio_prompts = batch['audio_features']
                        skeletons = batch['skeleton_frames']
                        width = batch['width']
                        height = batch['height']

                        # 移动到设备
                        ref_image = ref_image.to(self.accelerator.device, dtype=self.weight_dtype)
                        clip_image = clip_image.to(self.accelerator.device, dtype=self.weight_dtype)
                        video = video.to(self.accelerator.device, dtype=self.weight_dtype)
                        audio_prompts = audio_prompts.to(self.accelerator.device)
                        face_mask = face_mask.to(self.accelerator.device, dtype=self.weight_dtype)
                        skeletons = skeletons.to(self.accelerator.device, dtype=self.weight_dtype)

                        # 处理音频和图像特征
                        clip_image, audio_list_tensor, motion_buckets = self.audio_image_process(
                            audio_prompts,
                            clip_image,
                            self.config.num_train_frames
                        )

                        motion_buckets = motion_buckets.to(self.accelerator.device, dtype=self.weight_dtype)

                        # 预测噪声
                        noise_pred, noise = self.pred_noise(
                            ref_image,
                            clip_image,
                            video,
                            face_mask,
                            audio_list_tensor,
                            motion_buckets,
                            height,
                            width,
                            self.config.num_train_frames,
                            motion_bucket_scale=self.config.motion_bucket_scale,
                            controlnet_cond=skeletons,
                        )

                        # 计算损失
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                        # 检查损失是否为NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            if self.accelerator.is_main_process:
                                print(f"⚠️ 检测到NaN/Inf损失，跳过此步")
                            continue

                    # 反向传播
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        # 梯度裁剪
                        if self.config.enable_gradient_clipping:
                            self.accelerator.clip_grad_norm_(
                                [p for p in self.unet.parameters() if p.requires_grad] + 
                                [p for p in self.controlnet.parameters() if p.requires_grad],
                                max_norm=self.config.max_grad_norm
                            )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # 更新进度条
                        progress_bar.update(1)
                        global_step += 1
                        train_loss += loss.detach().item()

                        # 日志记录# 日志记录
                        if global_step % self.config.logging_steps == 0 and self.accelerator.is_main_process:
                            avg_loss = train_loss / self.config.logging_steps
                            logger.info(f"Step: {global_step}, Loss: {avg_loss:.4f}")
                            train_loss = 0.0

                        # 保存检查点
                        if global_step % self.config.save_steps == 0 and self.accelerator.is_main_process:
                            self._save_checkpoint(global_step, epoch)

                    logs = {
                        "loss": loss.detach().item(),
                        # "lr": self.lr_scheduler.get_last_lr()[0],
                        "ip_lr": self.lr_scheduler.get_last_lr()[0],
                        "controlnet_lr": self.lr_scheduler.get_last_lr()[1],
                        "step": global_step,
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step >= self.max_train_steps:
                        break

        # 保存最终模型
        if self.accelerator.is_main_process:
            self._save_checkpoint(global_step, final=True)
        self.accelerator.end_training()

    def whisper_process(self, audio_feature):
        """whisper音频处理"""

        # 去掉多余维度
        audio_feature = audio_feature.flatten(start_dim=0, end_dim=1)

        window = 3000
        # 确保音频特征是3000的整数倍
        total_length = audio_feature.shape[-1]
        if total_length % window != 0:
            # 补零到3000的整数倍
            pad_length = window - (total_length % window)
            padding = torch.zeros(*audio_feature.shape[:-1], pad_length,
                                  device=audio_feature.device, dtype=audio_feature.dtype)
            audio_feature = torch.cat([audio_feature, padding], dim=-1)

        audio_prompts = []
        wav_enc = self.whisper_model
        last_audio_prompts = []
        with torch.no_grad():
            for i in range(0, audio_feature.shape[-1], window):
                # seg = audio_feature[:, :, i:i + window]
                # # 隐藏层输出用于计算音频提示，音频提示获取音频的细粒度特征
                # audio_prompt = wav_enc.encoder(seg, output_hidden_states=True).hidden_states
                # # 最后一层输出用于运动桶计算
                # last_audio_prompt = wav_enc.encoder(seg).last_hidden_state
                seg = audio_feature[:, :, i:i + window]
                whisper_out = wav_enc.encoder(seg, output_hidden_states=True)
                # 隐藏层输出用于计算音频提示，音频提示获取音频的细粒度特征
                audio_prompt = whisper_out.hidden_states
                # 最后一层输出用于运动桶计算
                last_audio_prompt = whisper_out.last_hidden_state

                # last_audio_prompt -> [batch_size, time_steps, 1, hidden_dim]
                last_audio_prompt = last_audio_prompt.unsqueeze(-2)

                # audio_prompt -> [batch_size, time_steps, num_layers, hidden_dim]
                audio_prompt = torch.stack(audio_prompt, dim=2)

                audio_prompts.append(audio_prompt)
                last_audio_prompts.append(last_audio_prompt)

        audio_len = self.config.num_train_frames * 50 // self.config.fps

        # time_steps维度上拼接
        audio_prompts = torch.cat(audio_prompts, dim=1)[:, :audio_len, :, :]
        audio_prompts = torch.cat([
            torch.zeros_like(audio_prompts[:, :4]),
            audio_prompts,
            torch.zeros_like(audio_prompts[:, :6])
        ], 1)

        last_audio_prompts = torch.cat(last_audio_prompts, dim=1)[:, :audio_len, :, :]
        last_audio_prompts = torch.cat([
            torch.zeros_like(last_audio_prompts[:, :24]),
            last_audio_prompts,
            torch.zeros_like(last_audio_prompts[:, :26])
        ], 1)

        return audio_prompts, last_audio_prompts

    def audio_to_token_bucket(self, audio_prompts, last_audio_prompts, clip_image, num_frames):
        """图片提示、音频提示、运动桶参数计算"""
        with torch.no_grad():
            # (b, c, h, w) -> (b, 1024)
            image_embeds = self.clip_model(clip_image).image_embeds

        audio_tensor_list = []
        audio_tensor_for_bucket = []

        step = 1

        for i in range(num_frames):
            # (frame, bach_size, time_step, num_layers, hidden_dim)
            audio_clip = audio_prompts[:, i * 2 * step:i * 2 * step + 10].unsqueeze(0)
            # (frame, bach_size, time_step, 1, hidden_dim)
            audio_clip_for_bucket = last_audio_prompts[:, i * 2 * step:i * 2 * step + 50].unsqueeze(0)

            audio_tensor_list.append(audio_clip)
            audio_tensor_for_bucket.append(audio_clip_for_bucket)

        # (num_frames, bach_size, time_step, 5, hidden_dim)
        audio_tensor = torch.cat(audio_tensor_list, dim=0)
        # (num_frames, bach_size, time_step, 5, hidden_dim) -> (num_frames, bach_size, context_tokens, output_dim)
        audio_condition = self.audio2token(audio_tensor)
        # (num_frames, bach_size, context_tokens, output_dim) -> (bach_size, num_frames, context_tokens, output_dim)
        audio_condition = audio_condition.transpose(0, 1)

        # (num_frames, bach_size, time_step, 1, hidden_dim)
        bucket_tensor = torch.cat(audio_tensor_for_bucket, dim=0)
        # (batch_size, 1024) -> (batch_size * num_frames, 1024)
        image_for_bucket = image_embeds.repeat(num_frames, 1)
        # (num_frames, bach_size, time_step, 1, hidden_dim) -> (num_frames, bach_size, context_tokens, output_dim)
        motion_buckets = self.audio2bucket(bucket_tensor, image_for_bucket)
        # (num_frames, bach_size, context_tokens, output_dim) -> (bach_size, num_frames, context_tokens, output_dim)
        motion_buckets = motion_buckets.transpose(0, 1)
        # scaling
        motion_buckets = motion_buckets * 16 + 16

        return image_embeds, audio_condition, motion_buckets

    def audio_image_process(self, audio_feature, clip_image, num_train_frames):
        """
        音频、图片处理函数
        输入：音频采用特征、采样频率、参考图片、帧数
        输出：图片提示、音频提示、运动桶
        """
        audio_prompts, last_audio_prompts = self.whisper_process(audio_feature)
        image_embeds, audio_tensor_list, motion_buckets = self.audio_to_token_bucket(
            audio_prompts,
            last_audio_prompts,
            clip_image,
            num_train_frames
        )

        return image_embeds, audio_tensor_list, motion_buckets

    def pred_noise(self, ref_image, clip_image, video, face_mask, audio_prompts, motion_buckets,
                   height, width, num_train_frames, motion_bucket_scale, controlnet_cond=None):
        # 0. Default height and width to unet
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        height = height.tolist() or self.unet.sample_size * vae_scale_factor
        width = width.tolist() or self.unet.sample_size * vae_scale_factor

        self.check_inputs(ref_image, height, width)

        # (batch_size, 1024) -> (batch_size, 1, 1, 1024)
        clip_image = clip_image.unsqueeze(1).unsqueeze(1)
        # (batch_size, 1, 1, 1024) -> (batch_size, num_frames, seq_len, dim)
        clip_image = clip_image.repeat(1, num_train_frames, 1, 1)

        # (b, 4, height/8, width/8)
        image_latents = self._encode_image(ref_image)

        image_latents = image_latents.unsqueeze(1).repeat(1, num_train_frames, 1, 1, 1)

        motion_buckets = motion_buckets * motion_bucket_scale

        video_latents = self._encode_video(video)

        # 按元素重复，例：[1,2,3] -> [1,1,2,2,3,3]
        # (b, c, h, w) -> (b*f, c, h, w)
        face_mask = face_mask.repeat_interleave(num_train_frames, 0)

        # time_steps = torch.randint(
        #     0, len(self.train_scheduler.timesteps),
        #     (video_latents.shape[0],)
        # )
        # time_steps = self.train_scheduler.timesteps[time_steps].to(video_latents.device)
        # # sigmas = self.train_scheduler.sigmas[time_steps].to(video_latents.device)
        # video_latents = self.train_scheduler.scale_model_input(video_latents, time_steps)

        # 加噪
        time_steps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps,
            (video_latents.shape[0],),
            device=video_latents.device
        )

        noise = torch.randn_like(video_latents).to(dtype=self.weight_dtype, device=video_latents.device)

        # 确保所有张量在同一设备上
        video_latents = video_latents.to(device=video_latents.device, dtype=self.weight_dtype)
        noise = noise.to(device=video_latents.device, dtype=self.weight_dtype)
        time_steps = time_steps.to(device=video_latents.device)

        noisy_latents = self.train_scheduler.add_noise(video_latents, noise, time_steps)

        motion_buckets = torch.mean(motion_buckets, dim=1)
        motion_bucket_id = motion_buckets[:, 0].long()
        motion_bucket_id_exp = motion_buckets[:, 1].long()
        added_time_ids = self._get_add_time_ids(
            motion_bucket_id,
            motion_bucket_id_exp,
            clip_image.dtype,
        )
        added_time_ids = added_time_ids.to(device=self.accelerator.device, dtype=self.weight_dtype)

        # noisy_latents = self.noise_scheduler.scale_model_input(noisy_latents, time_steps)

        # (b, f, 4, h/8, w/8) (b, f, 4, h/8, w/8) ->(b, f, 8, h/8, w/8)
        latent_model_input = torch.cat(
            [noisy_latents, image_latents], dim=2
        )

        cross_attention_kwargs = {'ip_adapter_masks': [face_mask]}

        # ControlNet处理
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if controlnet_cond is not None:
            # 正常模式：ControlNet在GPU上
            # controlnet_device = self.controlnet.device
            # controlnet_dtype = self.controlnet.dtype

            controlnet_output = self.controlnet(
                latent_model_input.to(device=self.accelerator.device, dtype=self.weight_dtype),
                time_steps,
                encoder_hidden_states=audio_prompts.flatten(0, 1).to(device=self.accelerator.device, dtype=self.weight_dtype),
                added_time_ids=added_time_ids.to(device=self.accelerator.device, dtype=self.weight_dtype),
                controlnet_cond=controlnet_cond.to(device=self.accelerator.device, dtype=self.weight_dtype),
                conditioning_scale=self.config.controlnet_scale,
                return_dict=False,
            )
            down_block_additional_residuals = controlnet_output[0]  # 获取down block残差
            mid_block_additional_residual = controlnet_output[1]    # 获取mid block残差

        noise_pred = self.unet(
            latent_model_input,
            time_steps,
            encoder_hidden_states=(clip_image.flatten(0, 1), [audio_prompts.flatten(0, 1)]),
            cross_attention_kwargs=cross_attention_kwargs,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=False,
        )[0]

        if self.train_scheduler.config.prediction_type == "v_prediction":
            target = self.train_scheduler.get_velocity(video_latents, noise, time_steps)
        else:
            target = noise

        return noise_pred, target

    def _get_add_time_ids(
            self,
            motion_bucket_id,
            noise_aug_strength,
            dtype,
    ):
        fps = torch.tensor([self.config.fps]).unsqueeze(0).repeat(motion_bucket_id.shape[0], 1).to(motion_bucket_id.device)
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        # passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        passed_add_embed_dim =256 * len(add_time_ids)

        # 使用 unwrap_model 获取原始模型
        unet_model = self.accelerator.unwrap_model(self.unet)
        expected_add_embed_dim = unet_model.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.cat(add_time_ids, dim=1).to(dtype=dtype)

        return add_time_ids

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if isinstance(height, int) and isinstance(width, int):
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        elif isinstance(height, list) and isinstance(width, list):
            for height, width in zip(height, width):
                if height % 8 != 0 or width % 8 != 0:
                    raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        else:
            raise ValueError(f"`height` and `width` have to be int or list, but got {type(height)} and {type(width)}.")

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """编码视频到潜在空间"""
        b, t, c, h, w = video.shape
        video = rearrange(video, 'b t c h w -> (b t) c h w')

        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()
            latents = latents * 0.18215

        latents = rearrange(latents, '(b t) c h w -> b t c h w', b=b, t=t)
        return latents

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像到潜在空间"""
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    def _apply_conditioning_dropout(self,
                                    audio_features: torch.Tensor,
                                    clip_features: torch.Tensor,
                                    batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用条件丢弃"""
        # 随机丢弃音频
        if np.random.random() < self.config.audio_drop_prob:
            audio_features = torch.zeros_like(audio_features)

        # 随机丢弃图像
        if np.random.random() < self.config.image_drop_prob:
            clip_features = torch.zeros_like(clip_features)

        # 随机丢弃两者
        if np.random.random() < self.config.both_drop_prob:
            audio_features = torch.zeros_like(audio_features)
            clip_features = torch.zeros_like(clip_features)

        return audio_features, clip_features

    def _save_checkpoint(self, global_step: int, epoch: int = None, final: bool = False):
        """保存检查点"""
        if not self.accelerator.is_main_process:
            return

        save_path = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
        if final:
            save_path = os.path.join(self.config.output_dir, "final_model")

        os.makedirs(save_path, exist_ok=True)

        # 保存模型
        models_to_save = {
            "unet": self.accelerator.unwrap_model(self.unet),
            "controlnet": self.accelerator.unwrap_model(self.controlnet),
        }

        for model_name, model in models_to_save.items():
            model_dir = os.path.join(save_path, model_name)
            os.makedirs(model_dir, exist_ok=True)

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, "pytorch_model.bin")
            )

            if hasattr(model, "config"):
                with open(os.path.join(model_dir, "config.json"), "w") as f:
                    config_dict = dict(model.config)
                    json.dump(config_dict, f)

        logger.info(f"Saved checkpoint at step {global_step} to: {save_path}")


# ==================== 主函数 ====================
def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Train Sonic model")

    parser.add_argument("--memory_optimization", type=str, default="conservative", 
                       choices=["conservative", "balanced", "aggressive", "deepspeed", "fp16_safe", "alternating"],
                       help="显存优化策略")
    parser.add_argument("--enable_gradient_clipping", action="store_true", default=True,
                       help="是否启用梯度裁剪")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                       help="DeepSpeed配置文件路径")

    args = parser.parse_args()

    # 创建配置
    config = TrainingConfig(**vars(args))
    # config.data_root = ["data", "data_vfhq"]  # 替换为实际数据路径
    config.data_root = ["data_vfhq"]  # 替换为实际数据路径

    # 添加显存优化配置
    config.memory_optimization = args.memory_optimization
    config.enable_gradient_clipping = args.enable_gradient_clipping
    config.deepspeed_config = args.deepspeed_config

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 初始化训练器
    trainer = SonicTrainer(config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
