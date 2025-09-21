from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class InferenceConfig:
    """ Inference configuration """
    # 输入输出配置
    input_video_path: str = ""  # 输入视频路径
    input_audio_path: str = ""  # 输入音频路径
    output_dir: str = ""  # 输出目录
    output_filename: str = "generated_video.mp4"  # 输出文件名
    
    # 模型路径配置
    model_path: str = ""  # 主模型路径
    whisper_path: str = ""  # Whisper语音模型路径
    yolo_path: str = ''  # YOLO检测模型路径
    unet_checkpoint_path: str = ""  # UNet检查点路径
    controlnet_model_name_or_path: str = ""  # ControlNet模型路径
    
    # 生成参数配置
    num_inference_steps: int = 20  # 推理步数（减少可加速生成）
    guidance_scale: float = 7.5  # 指导尺度
    controlnet_conditioning_scale: float = 1.0  # ControlNet控制强度
    ip_audio_scale: float = 1.0  # 音频条件强度
    motion_bucket_scale: float = 1.0  # 运动桶缩放比例
    
    # 视频处理配置
    num_frames: int = 25  # 生成帧数
    fps: float = 25  # 帧率
    resolution: int = 512  # 分辨率
    sampling_rate: int = 16000  # 音频采样率
    
    # 设备与性能配置
    device: str = "cuda"  # 设备：cuda, cpu, mps
    mixed_precision: str = "fp16"  # 混合精度：no, fp16, bf16
    torch_dtype: str = "fp16"  # torch数据类型：fp16, fp32, bf16
    enable_xformers: bool = True  # 是否启用xformers优化
    enable_vae_slicing: bool = True  # 是否启用VAE切片节省显存
    enable_vae_tiling: bool = False  # 是否启用VAE平铺
    
    # 种子配置
    seed: int = 42  # 随机种子
    deterministic: bool = True  # 是否确定性生成
    
    save_frames: bool = False  # 是否保存中间帧
    save_processed_audio: bool = False  # 是否保存处理后的音频
    video_codec: str = "libx264"  # 视频编码器
    video_preset: str = "medium"  # 视频编码预设
    audio_bitrate: str = "192k"  # 音频比特率
    
    batch_size: int = 1  # 批处理大小
    max_batch_size: int = 4  # 最大批处理大小（根据显存调整）

    debug_mode: bool = False  # 调试模式
    verbose: bool = True  # 详细输出
    save_intermediate_results: bool = False  # 是否保存中间结果
    
    scheduler_type: str = "DPMSolverMultistepScheduler"  # 调度器类型
    use_cached_features: bool = False  # 是否使用缓存的特征
    cache_dir: Optional[str] = None  # 缓存目录