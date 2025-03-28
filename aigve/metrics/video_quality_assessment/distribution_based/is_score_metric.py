import os
import json
from typing import Dict, Sequence
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

@METRICS.register_module()
class ISScore(BaseMetric):

    def __init__(
            self, 
            model_name: str = 'inception_v3', 
            input_shape: tuple = (299, 299, 3), 
            splits: int = 10,
            is_gpu: bool = True):
        super(ISScore, self).__init__()
        self.device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")
        self.splits = splits
        self.results = []

        if model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=True, transform_input=False, aux_logits=True)
            self.model.eval().to(self.device)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, -1, 1, 1)
        images = (images - mean) / std
        return images

    def compute_inception_features(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_tensor(images).to(self.device)
        with torch.no_grad():
            output = self.model(images)
            if isinstance(output, tuple):
                output = output[0]
        return output.cpu()

    def calculate_is(self, preds: np.ndarray) -> float:
        kl = preds * (np.log(preds + 1e-10) - np.log(np.expand_dims(np.mean(preds, axis=0), 0) + 1e-10))
        kl_mean = np.mean(np.sum(kl, axis=1))
        return float(np.exp(kl_mean))

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        _, gen_tensor_tuple, _, video_names_tuple = data_samples

        for gen_tensor, video_name in zip(gen_tensor_tuple, video_names_tuple):
            logits = self.compute_inception_features(gen_tensor)
            preds = torch.nn.functional.softmax(logits, dim=1).numpy()
            score = self.calculate_is(preds)

            self.results.append({
                "video_name": video_name,
                "IS_Score": score
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        scores = [item["IS_Score"] for item in self.results]
        mean_score = float(np.mean(scores))

        final_results = {
            "video_results": self.results,
            "IS_Mean_Score": mean_score
        }

        json_file_path = os.path.join(os.getcwd(), "is_results.json")
        with open(json_file_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)

        print(f"Final IS mean score: {mean_score:.4f} saved to {json_file_path}")

        return final_results
