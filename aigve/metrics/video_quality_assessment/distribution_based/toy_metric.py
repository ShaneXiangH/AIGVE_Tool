# Copyright (c) IFM Lab. All rights reserved.
# Deprecated
from typing import Dict, List, Optional, Sequence, Union
from mmengine.evaluator import BaseMetric
from core.registry import METRICS
from mmengine.logging import MMLogger
import torch
import torch.nn as nn
import numpy as np

from utils import add_git_submodule, submodule_exists
from .GSTVQA.TCSVT_Release.GVQA_Release.GVQA_Cross.cross_test import GSTVQA as GSTVQA_model
from scipy import stats
import h5py
# metric_path = '/metrics/video_quality_assessment/nn_based/gstvqa'

@METRICS.register_module()
class Toy(BaseMetric):
    """The Toy evaluation metric. 
    
    Args:
        collect_device (str): Device used for collecting results from workers.
            Options: 'cpu' and 'gpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            Default: None.
        metric_path (str): the file path of the metric 
        train_index (int): The specific model used. Details on: https://github.com/Baoliang93/GSTVQA/blob/main/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py#L162
        datainfo_path (str): the file path of the dataset
    """

    default_prefix: Optional[str] = 'llm_score'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None, 
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.criterion = nn.L1Loss().to(self.device)


    def process(self, data_batch: Sequence, data_samples: Sequence) -> None:
        """Toy process
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence): A batch of data from the dataloader.
            data_samples (Sequence): A batch of data samples that
                contain annotations and predictions.
        """
        print(f"Type of data_samples: {type(data_samples)}")
        print(f"Content of data_samples: {data_samples}")

        # result = dict()

        # features, length, label, mean_var,std_var,mean_mean,std_mean = data_samples
        # # # prompt_gt = data_sample['prompt_gt'] # str
        # # video_pd = data_sample['video_pd'] # torch.uint8(F, C, H, W)

        # features = torch.tensor(features[0]).unsqueeze(1)
        # length = torch.tensor(length[0])
        # label = torch.tensor(label[0])  # Extract the tensor if it's a single-element tuple
        # mean_var = torch.tensor(mean_var[0])
        # std_var = torch.tensor(std_var[0])
        # mean_mean = torch.tensor(mean_mean[0])
        # std_mean = torch.tensor(std_mean[0])

        # result['y_test'] = self.scale * label.item()

        # features = features.to(self.device).float()
        # label = label.to(self.device).float()
        # mean_var = mean_var.to(self.device).float()
        # std_var = std_var.to(self.device).float()
        # mean_mean = mean_mean.to(self.device).float()
        # std_mean = std_mean.to(self.device).float()

        # print(f"features shape: {features.shape}")   # torch.Size([500, 1, 2944])
        # print(f"length shape: {length.shape}")        # torch.Size([1])
        # print(f"labels shape: {label.shape}")        # torch.Size([1])
        # print(f"mean_var shape: {mean_var.shape}")   # torch.Size([1472])
        # print(f"std_var shape: {std_var.shape}")     # torch.Size([1472])
        # print(f"mean_mean shape: {mean_mean.shape}") # torch.Size([1472])
        # print(f"std_mean shape: {std_mean.shape}")   # torch.Size([1472])

        outputs = self.model(features, length.float(),mean_var,std_var,mean_mean,std_mean)
        result['y_pred'] = self.scale * outputs.item()
        result['loss'] = self.criterion(outputs, label).item()

        self.results.append(result)


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # metric_results = dict()
        # logger: MMLogger = MMLogger.get_current_instance()

        # assert len(self.test_index) == len(results)
        # test_loss = sum(result.get('loss', 0) for result in results) / len(results)
        # y_pred_np = np.zeros(len(self.test_index))
        # y_test_np = np.zeros(len(self.test_index))
        # for i, result in enumerate(results):
        #     y_pred_np[i] = result['y_pred']
        #     y_test_np[i] = result['y_test']

        # PLCC = stats.pearsonr(y_pred_np, y_test_np)[0]
        # SROCC = stats.spearmanr(y_pred_np, y_test_np)[0]
        # RMSE = np.sqrt(((y_pred_np-y_test_np) ** 2).mean())
        # KROCC = stats.stats.kendalltau(y_pred_np, y_test_np)[0]
        # print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
        #         .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        
        # metric_results['PLCC'] = PLCC
        # metric_results['SROCC'] = SROCC
        # metric_results['RMSE'] = RMSE
        # metric_results['KROCC'] = KROCC

        return metric_results



