import os
import argparse
from sonic import Sonic


class SonicVideoGenerator:
    def __init__(self, device_id=0):
        """
        初始化Sonic视频生成器

        Args:
            device_id: 设备ID，默认为0
        """
        self.pipe = Sonic(device_id)
        self.face_info = None

    def preprocess_face(self, image_path, expand_ratio=0.5):
        """
        预处理人脸检测

        Args:
            image_path: 输入图像路径
            expand_ratio: 人脸检测扩展比例

        Returns:
            dict: 人脸信息字典
        """
        self.face_info = self.pipe.preprocess(image_path, expand_ratio=expand_ratio)
        print(f"检测到人脸信息: {self.face_info}")
        return self.face_info

    def crop_image_if_needed(self, input_image_path, output_crop_path, crop_bbox):
        """
        如果需要，裁剪图像

        Args:
            input_image_path: 原始图像路径
            output_crop_path: 裁剪后图像保存路径
            crop_bbox: 裁剪边界框
        """
        self.pipe.crop_image(input_image_path, output_crop_path, crop_bbox)
        return output_crop_path

    def generate_video(self, image_path, audio_path, output_path,
                       min_resolution=512, inference_steps=25,
                       dynamic_scale=1.0, expand_ratio=0.5,
                       auto_crop=False, crop_output_path=None):
        """
        生成视频的主接口方法

        Args:
            image_path: 输入图像路径
            audio_path: 输入音频路径
            output_path: 输出视频路径
            min_resolution: 最小分辨率
            inference_steps: 推理步数
            dynamic_scale: 动态缩放比例
            expand_ratio: 人脸检测扩展比例
            auto_crop: 是否自动裁剪
            crop_output_path: 裁剪图像保存路径

        Returns:
            str: 生成的视频文件路径
        """
        # 预处理人脸检测
        face_info = self.preprocess_face(image_path, expand_ratio)

        if face_info['face_num'] <= 0:
            raise ValueError("未检测到人脸，无法生成视频")

        # 如果需要裁剪
        processed_image_path = image_path
        if auto_crop and crop_output_path:
            if not crop_output_path.endswith('.png'):
                crop_output_path += '.crop.png'
            processed_image_path = self.crop_image_if_needed(
                image_path, crop_output_path, face_info['crop_bbox']
            )
            print(f"图像已裁剪并保存至: {processed_image_path}")

        # 创建输出目录
        output_path = output_path.replace('.mp4', '_sonic.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.pipe.process(
            processed_image_path,
            audio_path,
            output_path,
            min_resolution=min_resolution,
            inference_steps=inference_steps,
            dynamic_scale=dynamic_scale
        )

        print(f"Sonic视频生成完成: {output_path}")
        return output_path