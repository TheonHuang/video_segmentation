import pixellib
import cv2
import numpy as np
import argparse
from pixellib.semantic import semantic_segmentation
 
parser = argparse.ArgumentParser(description='Segment video')
parser.add_argument('--input_video', help='path to input video ')
args = parser.parse_args()
 
segment_video=semantic_segmentation()
segment_video.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
segment_video.process_video_pascalvoc(args.input_video, overlay=True, frames_per_second=15, output_video_name = "output.mp4")
