import os
import cv2
import numpy as np
import torch
import json
import sys
from pathlib import Path
from tqdm import tqdm

# 添加DWpose路径
sys.path.append('./DWPose')

# 导入DWpose相关模块
from dwpose.wholebody import Wholebody, HWC3, resize_image
from dwpose import util

def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    """绘制骨骼图"""
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas

class PoseProcessor:
    def __init__(self, detection_model_path, pose_model_path, device="cuda"):
        """初始化姿态处理器"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化DWpose模型
        self.pose_estimation = Wholebody(detection_model_path, pose_model_path, device=self.device)
        self.resize_size = 1024
        
    # @torch.no_grad()
    # def process_single_image(self, image):
    #     """处理单张图片"""
    #     # 预处理
    #     input_image = HWC3(image[..., ::-1])
    #     resized_image = resize_image(input_image, self.resize_size)

    #     # 推理
    #     candidate, subset, det_result = self.pose_estimation(resized_image)

    #     # 后处理
    #     H, W, C = resized_image.shape
    #     nums, keys, locs = candidate.shape

    #     # 归一化坐标
    #     candidate[..., 0] /= float(W)
    #     candidate[..., 1] /= float(H)

    #     # 默认选择最大人体目标
    #     max_area = 0
    #     max_index = 0

    #     for i in range(nums):
    #         # 从subset获取身体关键点置信度（前18个）
    #         body_conf = subset[i, :18]  # 形状 (18,)

    #         # 计算有效关键点数量
    #         valid_count = (body_conf > 0.3).sum()
    #         if valid_count < 5:  # 至少需要5个有效关键点
    #             continue

    #         # 获取当前人体的所有关键点（包括x,y）
    #         body_kpts = candidate[i, :18]  # 形状 (18, 2)

    #         # 使用subset置信度筛选有效关键点
    #         valid_mask = body_conf > 0.3
    #         valid_kpts = body_kpts[valid_mask]

    #         # 计算包围盒面积
    #         x_min, x_max = valid_kpts[:, 0].min(), valid_kpts[:, 0].max()
    #         y_min, y_max = valid_kpts[:, 1].min(), valid_kpts[:, 1].max()
    #         area = (x_max - x_min) * (y_max - y_min)

    #         if area > max_area:
    #             max_area = area
    #             max_index = i

    #     if max_area == 0:
    #         return None, None, (0, 0)

    #     # 只保留最大人体的关键点
    #     candidate = candidate[max_index:max_index + 1]
    #     subset = subset[max_index:max_index + 1]

    #     # 提取身体关键点
    #     body = candidate[:, :18].copy()
    #     body = body.reshape(1 * 18, locs)  # 修改为只处理一个人体
    #     score = subset[:, :18]

    #     # 处理置信度
    #     for i in range(len(score)):
    #         for j in range(len(score[i])):
    #             if score[i][j] > 0.3:
    #                 score[i][j] = int(18 * i + j)
    #             else:
    #                 score[i][j] = -1

    #     un_visible = subset < 0.3
    #     candidate[un_visible] = -1

    #     # 提取其他关键点（只保留最大人体的）
    #     foot = candidate[:, 18:24]
    #     faces = candidate[:, 24:92]
    #     hands = candidate[:, 92:113]
    #     hands = np.vstack([hands, candidate[:, 113:]])

    #     bodies = dict(candidate=body, subset=score)
    #     pose = dict(bodies=bodies, hands=hands, faces=faces)

    #     return pose, det_result[max_index] if det_result is not None else None, (H, W)


    @torch.no_grad()
    def process_single_image(self, image):
        """处理单张图片"""
        # 预处理
        input_image = HWC3(image[..., ::-1])
        resized_image = resize_image(input_image, self.resize_size)

        # 推理
        candidate, subset, det_result = self.pose_estimation(resized_image)
        H, W, C = resized_image.shape
        if 0 in candidate.shape or 0 in subset.shape or 0 in det_result.shape:
            return None, None, (H, W)
        
        # print(f"############################ Detected shape ###########################")
        # print(f"candidate shape: {candidate.shape}, subset shape: {subset.shape}, det_result shape: {det_result.shape if det_result is not None else None}")

        # 后处理
        nums, keys, locs = candidate.shape

        # 归一化坐标
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        # 默认选择最大人体目标
        max_area = 0
        max_index = -1

        for i in range(nums):

            # 直接从subset获取身体关键点置信度（前18个）
            body_conf = subset[i, :18]  # 形状 (18,)

            # 计算有效关键点数量
            valid_count = (body_conf > 0.3).sum()
            if valid_count < 5:  # 至少需要5个有效关键点
                continue

            # 获取当前人体的所有关键点（包括x,y）
            body_kpts = candidate[i, :18]  # 形状 (18, 2)

            # 使用subset置信度筛选有效关键点
            valid_mask = body_conf > 0.3
            valid_kpts = body_kpts[valid_mask]

            # 计算包围盒面积
            x_min, x_max = valid_kpts[:, 0].min(), valid_kpts[:, 0].max()
            y_min, y_max = valid_kpts[:, 1].min(), valid_kpts[:, 1].max()
            area = (x_max - x_min) * (y_max - y_min)

            if area > max_area:
                max_area = area
                max_index = i

        if max_index == -1:
            return None, None, (H, W)

        # 只保留最大人体的关键点
        candidate = candidate[max_index:max_index + 1]
        subset = subset[max_index:max_index + 1]

        # 提取身体关键点
        body = candidate[:, :18].copy()
        body = body.reshape(1 * 18, locs)  # 修改为只处理一个人体
        score = subset[:, :18]

        # 处理置信度
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18 * i + j)
                else:
                    score[i][j] = -1

        un_visible = subset < 0.3
        candidate[un_visible] = -1

        # 提取其他关键点（只保留最大人体的）
        foot = candidate[:, 18:24]
        faces = candidate[:, 24:92]
        hands = candidate[:, 92:113]
        hands = np.vstack([hands, candidate[:, 113:]])

        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        return pose, det_result[max_index], (H, W)

    def process_frames_batch(self, frames_dir, output_dir, output_json_path, batch_size=8):
        """批量处理帧序列并保存为图片序列"""
        # 获取所有帧文件
        frame_files = sorted([f for f in os.listdir(frames_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        if not frame_files:
            print("No image files found!")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取第一帧获取尺寸信息
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        if first_frame is None:
            print("Cannot read first frame!")
            return

        frame_height, frame_width = first_frame.shape[:2]

        keypoints_list = []
        processed_count = 0

        # 批量处理
        for i in tqdm(range(0, len(frame_files), batch_size), desc="Processing frames", leave=False):
            batch_files = frame_files[i:i + batch_size]

            for fname in batch_files:
                frame_path = os.path.join(frames_dir, fname)
                frame = cv2.imread(frame_path)

                if frame is None:
                    print(f"Warning: Cannot read frame {fname}")
                    keypoints_list.append(None)
                    continue

                try:
                    # 处理单帧
                    pose, det_result, (H, W) = self.process_single_image(frame)

                    # 如果没有检测到人体，保存空白图像
                    if pose is None:
                        # skeleton_img = np.zeros((H, W, 3), dtype=np.uint8)
                        skeleton_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                        keypoints_list.append(None)
                    else:
                        # 生成骨骼图
                        skeleton_img = draw_pose(pose, H, W, use_body=True, use_face=True)

                        # 保存关键点
                        if det_result is not None and len(det_result) > 0:
                            w_ratio, h_ratio = frame_width / W, frame_height / H
                            det_result[..., ::2] *= h_ratio
                            det_result[..., 1::2] *= w_ratio
                            det_result = det_result.astype(np.int32)
                            keypoints_list.append(det_result.tolist())
                        else:
                            keypoints_list.append(None)

                    # 调整回原始尺寸并转换颜色空间
                    skeleton_img = cv2.resize(skeleton_img[..., ::-1], (frame_width, frame_height),
                                              interpolation=cv2.INTER_LANCZOS4 if frame_height * frame_width > H * W else cv2.INTER_AREA)

                    # 保存骨骼图
                    output_path = os.path.join(output_dir, fname)
                    cv2.imwrite(output_path, skeleton_img)

                    processed_count += 1

                except Exception as e:
                    print(f"Error processing frame {fname}: {e}")
                    keypoints_list.append(None)

        # 保存关键点json
        with open(output_json_path, "w", encoding='utf-8') as f:
            json.dump(keypoints_list, f, ensure_ascii=False, indent=2)

def is_video_processed(video_dir, video_name):
    """
    验证视频文件是否已完整处理
    
    参数:
        video_dir (str): 视频目录路径
        video_name (str): 视频名称
    
    返回:
        tuple: (是否已处理, frames数量, skeletons数量)
    """
    frames_dir = os.path.join(video_dir, "frames")
    skeletons_dir = os.path.join(video_dir, "skeletons")
    json_path = os.path.join(video_dir, f"{video_name}_keypoints.json")
    
    # 检查关键文件是否存在
    required_files_exist = (
        os.path.exists(skeletons_dir) and 
        os.path.exists(json_path) and 
        os.path.exists(frames_dir)
        )
    
    if not required_files_exist:
        return False, 0, 0
    
    try:
        # 统计有效图片文件数量
        frame_files = [f for f in os.listdir(frames_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        skeleton_files = [f for f in os.listdir(skeletons_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 检查数量是否一致且不为空
        is_complete = (
            len(frame_files) > 0 and 
            len(frame_files) == len(skeleton_files))
        
        return is_complete, len(frame_files), len(skeleton_files)
    
    except Exception as e:
        print(f"验证视频 {video_name} 时出错: {str(e)}")
        return False, 0, 0

def verify_dataset(data_dir, video_dirs):
    """
    验证整个数据集的处理完整性
    
    参数:
        data_dir (str): 数据集根目录
        video_dirs (list): 视频目录列表
    
    返回:
        tuple: (缺失视频列表, 帧数不匹配视频列表)
    """
    missing_videos = []
    frame_mismatch_videos = []
    
    for video_name in video_dirs:
        video_dir = os.path.join(data_dir, video_name)
        is_processed, frame_count, skeleton_count = is_video_processed(video_dir, video_name)
        
        if frame_count != skeleton_count:
            frame_mismatch_videos.append((video_name, frame_count, skeleton_count))
        elif not is_processed:
            missing_videos.append(video_name)
    
    return missing_videos, frame_mismatch_videos

def main():
    """主函数"""
    # 导入配置
    import pose_config
    
    # 配置路径
    detection_model_path = pose_config.DETECTION_MODEL_PATH
    pose_model_path = pose_config.POSE_MODEL_PATH
    data_dir = "result_data"  # 数据根目录
    
    # 检查模型文件
    if not os.path.exists(detection_model_path):
        print(f"Detection model not found: {detection_model_path}")
        return
        
    if not os.path.exists(pose_model_path):
        print(f"Pose model not found: {pose_model_path}")
        return
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # 获取视频目录列表
    video_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    video_dirs = sorted(list(set(video_dirs)))

    # 取后三分之一的视频文件
    # start_index = len(video_dirs) // 2  # 计算三分之二处作为起始点
    # video_dirs = video_dirs[-4300:]  # 从起始点到末尾

    if not video_dirs:
        print("ℹ No valid video directories found in data folder")
        return
    
    print(f"🔍 Found {len(video_dirs)} videos to process")
    
    # 初始化处理器
    processor = PoseProcessor(detection_model_path, pose_model_path)

    # 统计变量
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # 使用tqdm显示进度条
    for video_name in tqdm(video_dirs, desc="Processing videos"):
        video_dir = os.path.join(data_dir, video_name)
        frames_dir = os.path.join(video_dir, "frames")
        skeletons_dir = os.path.join(video_dir, "skeletons")
        json_path = os.path.join(video_dir, f"{video_name}_keypoints.json")
        
        # 使用验证函数检查是否已处理
        is_processed, _, _ = is_video_processed(video_dir, video_name)
        if is_processed:
            tqdm.write(f"ℹ {video_name} already processed, skipping...")
            skipped_count += 1
            continue
        
        try:
            # 确保输出目录存在
            os.makedirs(skeletons_dir, exist_ok=True)
            
            # 处理当前视频的帧
            processor.process_frames_batch(frames_dir, skeletons_dir, json_path, batch_size=8)
            processed_count += 1
            tqdm.write(f"✔ Successfully processed {video_name}")
            
        except Exception as e:
            failed_count += 1
            tqdm.write(f"⚠ Failed to process {video_name}: {str(e)}")
            continue
    
    # 验证处理结果
    print("\n🔍 Verifying processing results...")
    missing_videos, frame_mismatch_videos = verify_dataset(data_dir, video_dirs)
    
    # 输出统计信息
    print("\n📊 Processing Statistics:")
    print(f"Total videos found: {len(video_dirs)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Failed to process: {failed_count}")
    
    if missing_videos:
        print("\n❌ Missing skeleton data for following videos:")
        for video in missing_videos:
            print(f"- {video}")
    else:
        print("\n✅ All videos have skeleton data!")
    
    if frame_mismatch_videos:
        print("\n⚠ Frame count mismatch in following videos (frames_dir vs skeletons_dir):")
        for video, frame_count, skeleton_count in frame_mismatch_videos:
            print(f"- {video}: {frame_count} frames vs {skeleton_count} skeletons")
    else:
        print("\n✅ All videos have matching frame counts in frames and skeletons directories!")

if __name__ == "__main__":
    main() 