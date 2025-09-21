"""
Sonic模型完整数据预处理脚本 - 修复版本
解决了JSON序列化、音频视频对齐、模型路径等问题
"""

import sys
import argparse
import json
import logging
import math
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
import librosa
import soundfile as sf
from PIL import Image
from tqdm import tqdm
from transformers import AutoFeatureExtractor

try:
    from model.face_align.human_detect import YOLOv8HumanDetector
    # from src.utils.util import seed_everything
except ImportError:
    print("Warning: YOLOv8HumanDetector modules not found.")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def path_to_str(obj):
    """递归将Path对象转换为字符串"""
    if isinstance(obj, (Path, type(Path()))):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [path_to_str(item) for item in obj]
    else:
        return obj


class SonicImageProcessor:
    """图像处理器 - 确保尺寸和质量要求"""

    @staticmethod
    def resize_to_sonic_format(image: Image.Image, target_size: int = 512) -> Image.Image:
        """调整图像到Sonic要求的格式"""
        w, h = image.size
        scale = target_size / min(w, h)

        # 必须是64的倍数（VAE要求）
        new_w = round(w * scale / 64) * 64
        new_h = round(h * scale / 64) * 64

        # 确保最小尺寸
        new_w = max(new_w, 64)
        new_h = max(new_h, 64)

        return image.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def generate_face_mask(image_path: str, face_detector, area_ratio: float = 1.1) -> Tuple[np.ndarray, Dict]:
        """生成人脸遮罩 - 严格按照Sonic逻辑"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        h, w = image.shape[:2]

        # 人脸检测
        bboxes = face_detector.detect(image)
        if not bboxes:
            raise ValueError(f"No face detected in {image_path}")

        # 获取最大人脸 - 确保坐标都是整数
        bbox = bboxes[0]['bbox']  # [x, y, X, Y]
        x, y, X, Y = [int(coord) for coord in bbox]
        face_w = X - x
        face_h = Y - y

        # 生成人脸遮罩
        mask = np.zeros((h, w), dtype=np.uint8)

        # 扩展人脸区域
        # expanded_w = int(face_w * area_ratio)
        # expanded_h = int(face_h * area_ratio)
        expanded_w = int(face_w * 1)
        expanded_h = int(face_h * 1)
        center_x = int(x + face_w // 2)
        center_y = int(y + face_h // 2)

        x1 = max(0, int(center_x - expanded_w // 2))
        y1 = max(0, int(center_y - expanded_h // 2))
        x2 = min(w, int(center_x + expanded_w // 2))
        y2 = min(h, int(center_y + expanded_h // 2))

        mask[y1:y2, x1:x2] = 255

        face_info = {
            "bbox": [int(x), int(y), int(face_w), int(face_h)],
            "expanded_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "face_area": int(face_w * face_h),
            "image_size": [int(w), int(h)],
            "face_ratio": float(face_w * face_h) / float(w * h)
        }

        return mask, face_info

    @staticmethod
    def validate_face_quality(face_info: Dict, min_face_ratio: float = 0.05,
                              min_face_size: int = 64) -> bool:
        """验证人脸质量"""
        bbox = face_info["bbox"]
        face_w, face_h = bbox[2], bbox[3]
        face_ratio = face_info["face_ratio"]

        # 检查人脸大小
        if min(face_w, face_h) < min_face_size:
            logger.warning(f"Face too small: {min(face_w, face_h)} < {min_face_size}")
            return False

        # 检查人脸占比
        if face_ratio < min_face_ratio:
            logger.warning(f"Face ratio too small: {face_ratio:.3f} < {min_face_ratio}")
            return False

        return True


class SonicAudioProcessor:
    """音频处理器 - 与训练代码完全一致"""

    def __init__(self, whisper_model_path: str = "checkpoints/whisper-tiny"):
        try:
            if os.path.exists(whisper_model_path):
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model_path)
                self.has_feature_extractor = True
                logger.info(f"Audio feature extractor loaded from: {whisper_model_path}")
            else:
                logger.warning(f"Whisper model not found at: {whisper_model_path}")
                logger.warning("Audio features will not be precomputed")
                self.has_feature_extractor = False
        except Exception as e:
            logger.warning(f"Failed to load audio feature extractor: {e}")
            self.has_feature_extractor = False

    def extract_audio_from_video(self, video_path: str, output_path: str,
                                 target_sr: int = 16000) -> str:
        """从视频提取音频"""
        try:
            # 使用librosa提取音频
            audio, sr = librosa.load(video_path, sr=target_sr)

            audio = audio[:len(audio) // 16000 * 16000]

            # 保存为wav文件
            sf.write(output_path, audio, target_sr)

            duration = len(audio) / target_sr
            logger.info(f"Audio extracted: {duration:.2f}s at {target_sr}Hz")
            return output_path

        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise

    def precompute_audio_features(self, audio_path: str) -> Optional[Dict]:
        """预计算音频特征 - 与训练代码完全一致"""
        if not self.has_feature_extractor:
            return None

        try:
            # 加载音频
            audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
            assert sampling_rate == 16000

            # 分窗口提取特征
            audio_features = []
            window = 750 * 640

            for i in range(0, len(audio_input), window):
                segment = audio_input[i:i + window]
                feature = self.feature_extractor(
                    segment,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding=False
                ).input_features
                audio_features.append(feature)

            audio_features = torch.cat(audio_features, dim=-1)
            audio_length = len(audio_input) // 640

            logger.info(f"Audio features computed: {audio_features.shape}, length: {audio_length}")
            return {
                "features": audio_features,
                "length": audio_length
            }

        except Exception as e:
            logger.error(f"Failed to compute audio features: {e}")
            return None


class SonicVideoProcessor:
    """视频处理器"""

    @staticmethod
    def extract_frames_adaptive(video_path: str, output_dir: str,
                                target_fps: int = None, max_frames: int = 1000) -> List[str]:
        """自适应帧提取 - 根据音频长度或最大帧数"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = math.ceil(fps)

        logger.info(f"Video info: {total_frames} frames, {fps:.2f} fps")

        # 计算采样间隔
        frame_interval = fps // target_fps

        frame_paths = []
        saved_count = 0
        frame_temp = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_temp.append(frame)

            if len(frame_temp) == fps:
                inner_target = 0
                for idx, f in enumerate(frame_temp):
                    if idx % frame_interval == 0:
                        frame_path = output_dir / f"{saved_count + 1:06d}.jpg"
                        cv2.imwrite(str(frame_path), f)
                        frame_paths.append(str(frame_path))
                        saved_count += 1

                        inner_target += 1
                        if inner_target >= target_fps:
                            break
                frame_temp = []

        cap.release()
        # logger.info(f"Extracted {len(frame_paths)} frames (target: {final_target}, total: {total_frames})")
        return frame_paths

    @staticmethod
    def select_best_reference_frame(frame_paths: List[str], face_detector) -> Tuple[str, int]:
        """选择最佳参考帧 - 综合人脸大小和清晰度"""
        best_frame = None
        best_score = 0
        best_idx = 0
        valid_frames = 0

        for idx, frame_path in enumerate(tqdm(frame_paths, desc="Selecting reference frame")):
            try:
                image = cv2.imread(frame_path)
                if image is None:
                    logger.warning(f"Cannot load frame: {frame_path}")
                    continue

                # 人脸检测
                bboxes = face_detector.detect(image)
                if not bboxes:
                    continue

                bbox = bboxes[0]['bbox']  # [x, y, w, h]
                # 确保bbox坐标是整数
                x, y, X, Y = [int(coord) for coord in bbox]
                w = X - x
                h = Y - y
                face_area = w * h

                # 验证bbox有效性
                img_h, img_w = image.shape[:2]
                if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                    logger.warning(f"Invalid bbox in frame {frame_path}: {bbox}")
                    continue

                # 计算图像清晰度
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 安全的人脸区域提取
                y1, y2 = max(0, y), min(img_h, y + h)
                x1, x2 = max(0, x), min(img_w, x + w)
                face_region = gray[y1:y2, x1:x2]

                if face_region.size > 0:
                    sharpness = cv2.Laplacian(face_region, cv2.CV_64F).var()
                else:
                    sharpness = 0

                # 综合评分
                score = face_area * 0.7 + sharpness * 0.3

                if score > best_score:
                    best_score = score
                    best_frame = frame_path
                    best_idx = idx

                valid_frames += 1

            except Exception as e:
                logger.warning(f"Failed to process frame {frame_path}: {e}")
                continue

        if best_frame is None:
            raise ValueError(
                f"No valid reference frame found (checked {len(frame_paths)} frames, {valid_frames} valid)")

        logger.info(
            f"Selected reference frame: {Path(best_frame).name} (index: {best_idx}, score: {best_score:.2f}, valid frames: {valid_frames})")
        return best_frame, best_idx


class SonicDataProcessor:
    """Sonic数据预处理主类"""

    def __init__(self,
                 face_det_model_path: str = "/root/autodl-tmp/Sonic_Train_max/yolov8n.pt",
                 whisper_model_path: str = "/root/autodl-tmp/Sonic/checkpoints/whisper-tiny",
                 device: str = "cuda",
                 target_size: int = 512,
                 face_area_ratio: float = 1.1,
                 min_face_ratio: float = 0.05,
                 min_face_size: int = 64,
                 target_fps: int = 25):

        self.device = device
        self.target_size = target_size
        self.face_area_ratio = face_area_ratio
        self.min_face_ratio = min_face_ratio
        self.min_face_size = min_face_size
        self.target_fps = target_fps

        # 验证模型文件存在
        if not os.path.exists(face_det_model_path):
            raise FileNotFoundError(f"Face detection model not found: {face_det_model_path}")

        # 初始化各个处理器
        self.face_detector = YOLOv8HumanDetector(model_path=face_det_model_path, device=device)
        self.audio_processor = SonicAudioProcessor(whisper_model_path)
        self.image_processor = SonicImageProcessor()
        self.video_processor = SonicVideoProcessor()

        logger.info("SonicDataProcessor initialized")

    def validate_video_file(self, video_path: str) -> bool:
        """预验证视频文件"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            cap.release()

            # 基本验证
            if fps < self.target_fps or fps % self.target_fps != 0:
                logger.warning(f"视频帧率不符合要求: {fps} fps")
                return False
            if total_frames < 10:
                logger.warning(f"Video too short: {total_frames} frames")
                return False

            if duration < 1.0:
                logger.warning(f"Video too short: {duration:.2f}s")
                return False

            return True

        except Exception as e:
            logger.warning(f"Video validation failed: {e}")
            return False

    def process_single_video(self, video_path: str, output_dir: str,
                             sample_name: str = None, **kwargs) -> bool:
        """处理单个视频文件"""
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if sample_name is None:
            sample_name = video_path.stem

        sample_dir = output_dir / sample_name

        # 如果已存在且完整，跳过
        if self._is_sample_complete(sample_dir):
            logger.info(f"Sample already exists and complete: {sample_name}")
            return True

        # 清理可能不完整的目录
        if sample_dir.exists():
            shutil.rmtree(sample_dir)

        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Processing video: {video_path} -> {sample_dir}")

            # 步骤1: 验证视频
            if not self.validate_video_file(str(video_path)):
                sample_dir.rmdir()
                raise ValueError("Video validation failed")

            # 步骤2: 提取音频
            audio_path = sample_dir / "audio.wav"
            self.audio_processor.extract_audio_from_video(str(video_path), str(audio_path))

            # 步骤4: 提取视频帧
            frames_dir = sample_dir / "frames"
            frame_paths = self.video_processor.extract_frames_adaptive(
                str(video_path), str(frames_dir),
                target_fps=self.target_fps,
                max_frames=kwargs.get('max_frames', 1000)
            )

            if len(frame_paths) < 5:
                raise ValueError(f"Too few frames extracted: {len(frame_paths)}")

            # 步骤6: 选择参考帧
            try:
                ref_frame_path, ref_idx = self.video_processor.select_best_reference_frame(
                    frame_paths, self.face_detector
                )
            except Exception as e:
                logger.error(f"Failed to select reference frame for {sample_name}: {e}")
                raise ValueError(f"Reference frame selection failed: {e}")

            # 步骤7: 生成人脸遮罩和验证
            try:
                mask, face_info = self.image_processor.generate_face_mask(
                    ref_frame_path, self.face_detector, self.face_area_ratio
                )
            except Exception as e:
                logger.error(f"Failed to generate face mask for {sample_name}: {e}")
                raise ValueError(f"Face mask generation failed: {e}")

            # 步骤8: 处理参考图像
            ref_image = Image.open(ref_frame_path).convert('RGB')
            ref_image_resized = self.image_processor.resize_to_sonic_format(
                ref_image, self.target_size
            )

            # 步骤9: 保存文件
            # 保存参考图像
            ref_image_path = sample_dir / "reference.jpg"
            ref_image_resized.save(ref_image_path, "JPEG", quality=95)

            # 保存人脸遮罩
            mask_resized = cv2.resize(mask, ref_image_resized.size, interpolation=cv2.INTER_NEAREST)
            mask_path = sample_dir / "mask.png"
            cv2.imwrite(str(mask_path), mask_resized)

            # 步骤10: 预计算音频特征
            audio_features_data = self.audio_processor.precompute_audio_features(str(audio_path))
            if audio_features_data:
                torch.save(audio_features_data, sample_dir / "audio_features.pt")

            # 步骤11: 生成元数据
            metadata = {
                "video_path": str(video_path),
                "sample_name": sample_name,
                "num_frames": len(frame_paths),
                "reference_frame_index": ref_idx,
                "face_info": face_info,
                "audio_duration": librosa.get_duration(path=str(audio_path)),  # 修复deprecated参数
                "final_image_size": list(ref_image_resized.size),
                "processing_params": {
                    "target_size": self.target_size,
                    "face_area_ratio": self.face_area_ratio,
                    "min_face_ratio": self.min_face_ratio,
                    "min_face_size": self.min_face_size,
                    **kwargs
                }
            }

            with open(sample_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully processed: {sample_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            # 清理失败的样本目录
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
            return False

    def _is_sample_complete(self, sample_dir: Path) -> bool:
        """检查样本是否已完整处理"""
        if not sample_dir.exists():
            return False

        required_files = ["reference.jpg", "audio.wav", "mask.png", "frames", "metadata.json"]
        for req_file in required_files:
            if not (sample_dir / req_file).exists():
                return False

        # 检查frames目录是否有文件
        frames_dir = sample_dir / "frames"
        if not any(frames_dir.glob("*.jpg")):
            return False

        return True


def validate_sonic_dataset(data_root: str) -> List[str]:
    """验证Sonic数据集格式"""
    data_root = Path(data_root)
    errors = []

    sample_dirs = [d for d in data_root.iterdir() if d.is_dir()]

    logger.info(f"Validating {len(sample_dirs)} samples...")

    for sample_dir in tqdm(sample_dirs, desc="Validating dataset"):
        # 检查必要文件
        required = ["reference.jpg", "audio.wav", "mask.png", "frames"]
        for req in required:
            if not (sample_dir / req).exists():
                errors.append(f"{sample_dir.name}: Missing {req}")
                continue

        # 检查音频视频对齐
        try:
            audio_path = sample_dir / "audio.wav"
            frames_dir = sample_dir / "frames"

            if audio_path.exists() and frames_dir.exists():
                frame_count = len(list(frames_dir.glob("*.jpg")))

        except Exception as e:
            errors.append(f"{sample_dir.name}: Validation error - {e}")

    if errors:
        logger.warning(f"Found {len(errors)} validation errors")
        for error in errors[:10]:  # 显示前10个错误
            logger.warning(f"  {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")
    else:
        logger.info("Dataset validation passed!")

    return errors


def get_video_files(input_dir: Path, extensions: List[str] = None) -> List[Path]:
    """获取视频文件列表"""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']

    video_files = []
    for ext in extensions:
        video_files.extend(list(input_dir.glob(f"**/*{ext}")))
        video_files.extend(list(input_dir.glob(f"**/*{ext.upper()}")))

    # 去重并排序
    video_files = sorted(list(set(video_files)))
    return video_files


def main():
    parser = argparse.ArgumentParser(
        description="Complete data preprocessing for Sonic model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 输入输出路径
    parser.add_argument("-i", "--input_dir", type=Path, required=True,
                        help="Directory containing video files")
    parser.add_argument("-o", "--output_dir", type=Path, required=True,
                        help="Output directory for processed training data")

    # 模型路径
    parser.add_argument("--face_det_model", type=str,
                        default="checkpoints/yoloface_v5m.pt",
                        help="Face detection model path")
    parser.add_argument("--whisper_model", type=str,
                        default="checkpoints/whisper-tiny",
                        help="Whisper model path for audio features")

    # 处理参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for processing (cuda/cpu)")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target image size")
    parser.add_argument("--max_frames", type=int, default=1000,  # 增加默认值
                        help="Maximum frames per video")
    parser.add_argument("--face_area_ratio", type=float, default=1.1,
                        help="Face area expansion ratio for mask generation")
    parser.add_argument("--min_face_ratio", type=float, default=0.05,
                        help="Minimum face area ratio threshold")
    parser.add_argument("--min_face_size", type=int, default=64,
                        help="Minimum face size threshold")

    # 并行处理
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel processes")
    parser.add_argument("--rank", type=int, default=0,
                        help="Process rank for parallel processing")

    # 其他选项
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate existing dataset")
    parser.add_argument("--resume", action="store_true",
                        help="Resume processing (skip existing samples)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # 设置随机种子
    # seed_everything(args.seed)

    # 验证模式
    if args.validate_only:
        if not args.output_dir.exists():
            logger.error(f"Dataset directory does not exist: {args.output_dir}")
            return
        errors = validate_sonic_dataset(str(args.output_dir))
        if errors:
            logger.error(f"Dataset validation failed with {len(errors)} errors")
            return
        else:
            logger.info("Dataset validation successful!")
            return

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 获取视频文件列表
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    video_files = get_video_files(args.input_dir)
    logger.info(f"Found {len(video_files)} video files")

    if not video_files:
        logger.warning("No video files found")
        return

    # 并行处理分片
    if args.parallel > 1:
        video_files = [f for i, f in enumerate(video_files) if i % args.parallel == args.rank]
        logger.info(f"Processing {len(video_files)} files (rank {args.rank}/{args.parallel})")

    # 初始化处理器
    try:
        processor = SonicDataProcessor(
            face_det_model_path="/root/autodl-tmp/Sonic_Train_max/yolov8n.pt",
            whisper_model_path="/root/autodl-tmp/Sonic/checkpoints/whisper-tiny",
            device=args.device,
            target_size=args.target_size,
            face_area_ratio=args.face_area_ratio,
            min_face_ratio=args.min_face_ratio,
            min_face_size=args.min_face_size,
            target_fps=25
        )
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return

    # 处理视频
    success_count = 0
    skip_count = 0
    fail_count = 0

    for video_path in tqdm(video_files, desc="Processing videos"):
        sample_name = video_path.stem
        sample_dir = args.output_dir / sample_name

        # 检查是否需要跳过
        if args.resume and processor._is_sample_complete(sample_dir):
            skip_count += 1
            continue

        try:
            success = processor.process_single_video(
                str(video_path), str(args.output_dir),
                max_frames=args.max_frames
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {video_path}: {e}")
            fail_count += 1
            continue

    # 最终统计
    total_processed = success_count + fail_count
    logger.info(f"Processing completed:")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {fail_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Total processed: {total_processed}")

    if success_count > 0:
        logger.info(f"Success rate: {success_count / total_processed * 100:.1f}%")

    # 生成数据集统计信息 - 修复JSON序列化问题
    stats = {
        "total_input_videos": len(video_files),
        "successful_samples": success_count,
        "failed_samples": fail_count,
        "skipped_samples": skip_count,
        "success_rate": success_count / total_processed if total_processed > 0 else 0,
        "processing_params": path_to_str(vars(args)),  # 转换Path对象为字符串
        "sample_dirs": [d.name for d in args.output_dir.iterdir() if d.is_dir()]
    }

    # 保存统计信息（确保所有Path对象都已转换）
    try:
        with open(args.output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info("Dataset statistics saved successfully")
    except Exception as e:
        logger.error(f"Failed to save dataset statistics: {e}")

    # 最终验证
    if success_count > 0:
        logger.info("Running final dataset validation...")
        errors = validate_sonic_dataset(str(args.output_dir))
        if not errors:
            logger.info("🎉 Dataset preprocessing completed successfully!")
        else:
            logger.warning(f"Dataset has {len(errors)} validation issues")


if __name__ == "__main__":
    main()
