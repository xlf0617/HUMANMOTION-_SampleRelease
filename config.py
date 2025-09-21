from dataclasses import dataclass, field
from typing import List
@dataclass
class TrainingConfig:
    """ Training configuration """
    # data_root: str = ["data","data_vfhq"]  # 数据集
    data_root: List[str] = field(default_factory=list)
    output_dir: str = ""  # 输出目录
    model_path: str = ""
    whisper_path: str = ""
    yolo_path: str = ''
    unet_checkpoint_path: str = ""
    controlnet_model_name_or_path: str = ""  

    train_batch_size: int = 8        # batch_size
    gradient_accumulation_steps: int = 16       # 梯度累积步数（减少以节省显存）
    num_train_epochs: int = 2      # epoch
    learning_rate: float = 1e-5      # 学习率（为FP16训练降低）
    ip_learning_rate: float = 5e-5
    controlnet_learning_rate: float = 1e-4
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"         # 混合精度
    weight_dtype: str = "fp16"  # "fp16", "fp32", "bf16"       # 权重类型
    max_grad_norm: float = 5.0       # 梯度剪裁（为FP16训练提高阈值）
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-3
    adam_epsilon: float = 1e-8
    lr_scheduler: str = "constant_with_warmup"  # 使用带预热的恒定学习率
    lr_warmup_steps: int = 50   # 增加预热步数

    num_workers: int = 8          # 数据加载线程数
    num_train_frames: int = 25      # 训练帧数（减少以节省显存）
    fps: float = 25               # fps
    resolution: int = 512            # 标准图片尺寸
    ip_audio_scale: float = 1.0      # 参考原文
    sampling_rate: int = 16000       # 音频采样频率
    motion_bucket_scale: float = 1.0  # 运动桶缩放比例
    controlnet_scale: float = 1.0     # ControlNet控制强度

    save_steps: int = 5000           # 保存检查点步数
    logging_steps: int = 200          # 保存日志步数
    validation_steps: int = 500      # 验证间隔步数
    use_wandb: bool = False
    wandb_project: str = "sonic"
    seed: int = 42                   # 随机种子

    audio_drop_prob: float = 0.1     # 音频提示弃用概率
    image_drop_prob: float = 0.1     # 图片提示弃用概率
    both_drop_prob: float = 0.1      # 音频、图片同时弃用概率
    controlnet_drop_prob: float = 0.1  # ControlNet条件弃用概率

    # 显存优化配置
    memory_optimization: str = "conservative"  # "conservative", "balanced", "aggressive", "deepspeed"
    enable_gradient_clipping: bool = True      # 是否启用梯度裁剪
    deepspeed_config: str = None               # DeepSpeed配置文件路径
