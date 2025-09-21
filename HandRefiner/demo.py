from __future__ import absolute_import, division, print_function
import sys
from config import handrefiner_root
import os

def load():
    paths = [handrefiner_root, os.path.join(handrefiner_root, 'MeshGraphormer'), os.path.join(handrefiner_root, 'preprocessor')]
    for p in paths:
        sys.path.insert(0, p)

load()

import cv2
import numpy as np
import torch
import os
import tempfile
import subprocess
from pathlib import Path
import argparse
from handrefiner import parse_args, model, meshgraphormer
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
import einops
from PIL import Image
import config
from cldm.model import create_model, load_state_dict
from preprocessor.meshgraphormer import MeshGraphormerMediapipe


class VideoHandRefiner:
    def __init__(self, strength=0.55, weights_path="models/inpaint_depth_control.ckpt",
                 prompt="a person making hand gestures", seed=1, padding_bbox=30):
        """
        初始化视频手部修复器

        Args:
            strength: ControlNet控制强度
            weights_path: 模型权重路径
            prompt: 修复提示词
            seed: 随机种子
            padding_bbox: 手部检测边界框填充
        """
        self.strength = strength
        self.prompt = prompt
        self.seed = seed
        self.padding_bbox = padding_bbox
        self.n_prompt = "fake 3D rendered image, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"

        # 确保模型已加载
        # self.model = model
        self.model = create_model("control_depth_inpaint.yaml").cpu()
        self.model.load_state_dict(
            load_state_dict("models/sd-v1-5-inpainting.ckpt", location="cuda"), strict=False
        )
        self.model.load_state_dict(
            load_state_dict("models/control_v11f1p_sd15_depth.pth", location="cuda"),
            strict=False,
        )
        self.model = model.to("cuda")

        # self.meshgraphormer = meshgraphormer
        self.meshgraphormer = MeshGraphormerMediapipe()
        self.ddim_sampler = DDIMSampler(self.model)

    def process_frame(self, frame, frame_count):
        """
        处理单帧图像

        Args:
            frame: 输入帧 (BGR格式)
            frame_count: 帧计数（用于命名）

        Returns:
            处理后的帧 (BGR格式)
        """
        # 转换BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, C = rgb_frame.shape

        # 使用meshgraphormer获取深度图和掩码
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存临时帧
            temp_frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(temp_frame_path, frame)

            # 获取深度图和掩码
            depthmap, mask, info = self.meshgraphormer.get_depth(
                temp_dir, f"frame_{frame_count}.jpg", self.padding_bbox
            )

        # 如果没有检测到手部，直接返回原帧
        if np.all(mask == 0):
            print(f"Frame {frame_count}: No hands detected, skipping processing")
            return frame

        # 预处理控制图像
        control = depthmap.astype(np.float32) / 255.0

        # 预处理源图像
        source = (rgb_frame.astype(np.float32) / 127.5) - 1.0
        source = source.transpose([2, 0, 1])  # source is c h w

        # 预处理掩码
        mask_processed = mask.astype(np.float32) / 255.0
        mask_processed = mask_processed[None]
        mask_processed[mask_processed < 0.5] = 0
        mask_processed[mask_processed >= 0.5] = 1

        # 创建掩码图像
        masked_image = source * (mask_processed < 0.5)  # masked image is c h w

        # 准备张量
        num_samples = 1
        mask_tensor = torch.stack([torch.tensor(mask_processed) for _ in range(num_samples)], dim=0).to("cuda")
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(64, 64))

        masked_image_tensor = torch.stack(
            [torch.tensor(masked_image) for _ in range(num_samples)], dim=0
        ).to("cuda")
        masked_image_tensor = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(masked_image_tensor)
        )

        x_tensor = torch.stack([torch.tensor(source) for _ in range(num_samples)], dim=0).to("cuda")
        z_tensor = self.model.get_first_stage_encoding(self.model.encode_first_stage(x_tensor))

        cats = torch.cat([mask_tensor, masked_image_tensor], dim=1)

        # 准备控制提示
        control_tensor = control[None, :].repeat(3, axis=0)
        control_tensor = torch.stack([torch.tensor(control_tensor) for _ in range(num_samples)], dim=0).to("cuda")

        # 创建条件
        cond = {
            "c_concat": [cats],
            "c_control": [control_tensor],
            "c_crossattn": [self.model.get_learned_conditioning([self.prompt] * num_samples)],
        }
        un_cond = {
            "c_concat": [cats],
            "c_control": [control_tensor],
            "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * num_samples)],
        }

        shape = (4, H // 8, W // 8)

        # 设置随机种子
        seed_everything(self.seed)

        # 设置控制强度
        self.model.control_scales = [self.strength] * 13

        # 采样
        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps=50,
            num_samples=num_samples,
            shape=shape,
            cond=cond,
            verbose=False,
            unconditional_guidance_scale=9.0,
            unconditional_conditioning=un_cond,
            x0=z_tensor,
            mask=mask_tensor
        )

        # 解码样本
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        # 转换回BGR
        result_frame = cv2.cvtColor(x_samples[0], cv2.COLOR_RGB2BGR)

        return result_frame

    # def process_video(self, input_video_path, output_video_path, process_every_n_frames=1):
    #     """
    #     处理整个视频
    #
    #     Args:
    #         input_video_path: 输入视频路径
    #         output_video_path: 输出视频路径
    #         process_every_n_frames: 每隔多少帧处理一次（1表示处理每一帧）
    #     """
    #     # 打开输入视频
    #     cap = cv2.VideoCapture(input_video_path)
    #     if not cap.isOpened():
    #         raise ValueError(f"无法打开视频文件: {input_video_path}")
    #
    #     # 获取视频属性
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    #     # 创建视频写入器
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    #
    #     print(f"开始处理视频: {input_video_path}")
    #     print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
    #     print(f"处理频率: 每{process_every_n_frames}帧处理一次")
    #
    #     frame_count = 0
    #     processed_count = 0
    #
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #
    #         # 决定是否处理当前帧
    #         if frame_count % process_every_n_frames == 0:
    #             print(f"处理帧 {frame_count}/{total_frames}")
    #             try:
    #                 processed_frame = self.process_frame(frame, frame_count)
    #                 processed_count += 1
    #             except Exception as e:
    #                 print(f"处理帧 {frame_count} 时出错: {e}")
    #                 processed_frame = frame  # 出错时使用原帧
    #         else:
    #             processed_frame = frame  # 跳过处理
    #
    #         # 写入处理后的帧
    #         out.write(processed_frame)
    #         frame_count += 1
    #
    #     # 释放资源
    #     cap.release()
    #     out.release()
    #
    #     print(f"视频处理完成!")
    #     print(f"总帧数: {frame_count}, 处理帧数: {processed_count}")
    #     print(f"输出视频已保存至: {output_video_path}")

    def process_video(self, input_video_path, output_video_path, process_every_n_frames=1):
        """
        处理整个视频并保留音频

        Args:
            input_video_path: 输入视频路径
            output_video_path: 输出视频路径
            process_every_n_frames: 每隔多少帧处理一次（1表示处理每一帧）
        """
        # 创建临时文件用于存储处理后的无音频视频
        temp_video_path = os.path.join(tempfile.gettempdir(), "temp_processed_video.mp4")

        # 打开输入视频
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_video_path}")

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建临时视频写入器（无音频）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        print(f"开始处理视频: {input_video_path}")
        print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
        print(f"处理频率: 每{process_every_n_frames}帧处理一次")

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 决定是否处理当前帧
            if frame_count % process_every_n_frames == 0:
                print(f"处理帧 {frame_count}/{total_frames}")
                try:
                    processed_frame = self.process_frame(frame, frame_count)
                    processed_count += 1
                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {e}")
                    processed_frame = frame  # 出错时使用原帧
            else:
                processed_frame = frame  # 跳过处理

            # 写入处理后的帧
            out.write(processed_frame)
            frame_count += 1

        # 释放资源
        cap.release()
        out.release()

        print(f"视频帧处理完成!")
        print(f"总帧数: {frame_count}, 处理帧数: {processed_count}")

        # 使用ffmpeg合并音频
        self._merge_audio_with_ffmpeg(input_video_path, temp_video_path, output_video_path)

        # 清理临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        print(f"最终视频已保存至: {output_video_path}")

    def _merge_audio_with_ffmpeg(self, input_video_path, processed_video_path, output_video_path):
        """
        使用ffmpeg合并处理后的视频和原始音频

        Args:
            input_video_path: 原始视频路径（包含音频）
            processed_video_path: 处理后的视频路径（无音频）
            output_video_path: 最终输出路径
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # 使用ffmpeg合并音频
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-i', processed_video_path,  # 处理后的视频
            '-i', input_video_path,  # 原始视频（用于提取音频）
            '-c:v', 'copy',  # 复制视频流
            '-c:a', 'aac',  # 编码音频为AAC
            '-map', '0:v:0',  # 使用第一个输入的视频流
            '-map', '1:a:0',  # 使用第二个输入的音频流
            '-shortest',  # 以较短的流为准
            output_video_path
        ]

        try:
            # 运行ffmpeg命令
            subprocess.run(cmd, check=True, capture_output=True)
            print("音频合并成功")
        except subprocess.CalledProcessError as e:
            print(f"音频合并失败: {e}")
            # 如果合并失败，至少保存处理后的视频
            if os.path.exists(processed_video_path):
                import shutil
                shutil.copy2(processed_video_path, output_video_path)
                print("已保存无音频版本的处理后视频")
        except FileNotFoundError:
            print("未找到ffmpeg，请安装ffmpeg以保留音频")
            # 如果没有ffmpeg，保存无音频版本
            import shutil
            shutil.copy2(processed_video_path, output_video_path)
            print("已保存无音频版本的处理后视频")


def main():
    parser = argparse.ArgumentParser(description='Process video to fix malformed hands')
    parser.add_argument('--input_video', type=str, required=True, help='Input video path')
    parser.add_argument('--output_video', type=str, required=True, help='Output video path')
    parser.add_argument('--strength', type=float, default=0.55, help='Control strength')
    parser.add_argument('--prompt', type=str, default="a person making hand gestures", help='Prompt for inpainting')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--padding', type=int, default=30, help='Padding for hand detection')
    parser.add_argument('--process_every', type=int, default=1, help='Process every N frames')

    args = parser.parse_args()

    # 初始化视频处理器
    refiner = VideoHandRefiner(
        strength=args.strength,
        prompt=args.prompt,
        seed=args.seed,
        padding_bbox=args.padding
    )

    # 处理视频
    refiner.process_video(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        process_every_n_frames=args.process_every
    )


if __name__ == "__main__":
    main()