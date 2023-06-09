import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Segment video')
parser.add_argument('--input_video', help='path to input video ')
args = parser.parse_args()
#import pixellib
#from pixellib.semantic import semantic_segmentation
 
# 创建一个实例并加载模型
#segment_video = semantic_segmentation()
#segment_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
 
# 使用 PixelLib 处理视频
output_video_name = "output.mp4"
#segment_video.process_video_pascalvoc("input.mp4", overlay=True, frames_per_second=15, output_video_name=output_video_name)

# 使用 OpenCV 读取处理过的视频
cap = cv2.VideoCapture(output_video_name)
cap_ori = cv2.VideoCapture(args.input_video)
# 获取视频的一些基本信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
 
# 创建一个 VideoWriter 对象
out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
 
kernel = np.ones((8,8), np.uint8)

# 循环处理每一帧
while cap.isOpened():
   ret, frame = cap.read()
   ret_ori, frame_ori = cap_ori.read()

   if not ret or not ret_ori:
       break
 
   # 将背景设置为白色
   white_background = np.ones_like(frame) * 255
 
   # 从分割的帧中获取人物的 mask
   person_mask = frame[..., 1] > 100
 
   person_mask = cv2.dilate(person_mask.astype(np.uint8),kernel, iterations = 2)
   person_mask =cv2.erode(person_mask, kernel, iterations = 2)
 
   # 使用 mask 将人物和白色背景合并
   result = np.where(person_mask[..., None], frame_ori, white_background)
 
   # 保存结果帧
   out.write(result.astype(np.uint8))
 
# 释放资源
cap.release()
cap_ori.release()
out.release()
