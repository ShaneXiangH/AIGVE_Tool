# Copyright (c) IFM Lab. All rights reserved.

from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.distribution_based.fid_metric import FIDScore
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
    type=FIDScore,
    model_name='inception_v3',  # The model used for FID calculation (commonly InceptionV3)
    input_shape=(299, 299, 3),  # Image input size for InceptionV3
    is_gpu = True
)
