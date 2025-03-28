# Copyright (c) IFM Lab. All rights reserved.

from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.distribution_based.fvd.fvd_metric import FVDScore
from datasets import FidDataset

with read_base():
    from ._base_.default import *


val_dataloader = dict(
    batch_size=2,  
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=FidDataset,
        video_dir = '/home/zhuosheng/VQA_Toolkit/aigve/data/toy/evaluate/',
        prompt_dir = '/home/zhuosheng/VQA_Toolkit/aigve/data/toy/annotations/evaluate.json', 
        max_len=20,
        if_pad=False
    )
)

val_evaluator = dict(
    type=FVDScore,
    model_path='AIGVE_Tool/aigve/metrics/video_quality_assessment/distribution_based/fvd/model_rgb.pth',  # Path to the I3D model
    feature_layer=-2,  # Feature layer to extract from the I3D model
    is_gpu=True
)
