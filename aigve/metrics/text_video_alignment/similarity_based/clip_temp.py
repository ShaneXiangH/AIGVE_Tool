# Copyright (c) IFM Lab. All rights reserved.

# code from https://evalcrafter.github.io
# encoding = utf-8

import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import time
import logging
# import wandb
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torchvision.utils import save_image


def calculate_clip_temp_score(video_path, model):
    ''' time score: frame[i] <-> frame[i+1] cosine similarity score
    '''
    # Load the video
    cap = cv2.VideoCapture(video_path)
    to_tensor = transforms.ToTensor()
    # Extract frames from the video 
    frames = []
    SD_images = []
    resize = transforms.Resize([224,224])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(frame)
    
    tensor_frames = torch.stack([resize(torch.from_numpy(frame).permute(2, 0, 1).float()) for frame in frames])

    # tensor_frames = [extracted_frames[i] for i in range(extracted_frames.size()[0])]
    concatenated_frame_features = []

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():  
        for frame in tensor_frames: # Too many frames in a video, must split before CLIP embedding, limited by the memory
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_feature = model.get_image_features(frame_input)
            concatenated_frame_features.append(frame_feature)

    concatenated_frame_features = torch.cat(concatenated_frame_features, dim=0)

    # Calculate the similarity scores
    clip_temp_score = []
    concatenated_frame_features = concatenated_frame_features / concatenated_frame_features.norm(p=2, dim=-1, keepdim=True)
    # ipdb.set_trace()

    for i in range(concatenated_frame_features.size()[0]-1):
        clip_temp_score.append(concatenated_frame_features[i].unsqueeze(0) @ concatenated_frame_features[i+1].unsqueeze(0).T)
    clip_temp_score=torch.cat(clip_temp_score, dim=0)
    # Calculate the average CLIP score across all frames, reflects temporal consistency 
    clip_temp_score_avg = clip_temp_score.mean().item()

    return clip_temp_score_avg


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='./video', help="Specify the path of generated videos")
    parser.add_argument("--dir_prompts", type=str, default='./prompts', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='clip_temp_score', help="Specify the metric to be used")
    args = parser.parse_args()

    dir_videos = args.dir_videos
    metric = args.metric

    dir_prompts =  args.dir_prompts
   
    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
    prompt_paths = [os.path.join(dir_prompts, os.path.splitext(os.path.basename(x))[0]+'.txt') for x in video_paths]

     # Create the directory if it doesn't exist
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"results", exist_ok=True)
    # Set up logging
    log_file_path = f"results/{metric}_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"results/{metric}_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model = CLIPModel.from_pretrained("checkpoints/clip-vit-base-patch32").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("checkpoints/clip-vit-base-patch32")
    
    # Calculate SD scores for all video-text pairs
    scores = []
    
    test_num = 10
    test_num = len(video_paths)
    count = 0
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        prompt_path = prompt_paths[i]
        if count == test_num:
            break
        else:
            text = read_text_file(prompt_path)
            # ipdb.set_trace()
            if metric == 'clip_temp_score':
                score = calculate_clip_temp_score(video_path,clip_model)
            else:
                raise ValueError('invalid metric choice')
            count+=1
            scores.append(score)
            average_score = sum(scores) / len(scores)
            # count+=1
            logging.info(f"Vid: {os.path.basename(video_path)},  Current {metric}: {score}, Current avg. {metric}: {average_score},  ")
            
    # Calculate the average SD score across all video-text pairs
    logging.info(f"Final average {metric}: {average_score}, Total videos: {len(scores)}")
