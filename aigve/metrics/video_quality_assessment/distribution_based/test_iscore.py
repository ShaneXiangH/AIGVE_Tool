import torch
from is_score_metric import ISScore  # 假设你将上述代码保存成 is_score.py 文件

def test_inception_score():
    num_images = 100
    fake_images = torch.rand((num_images, 3, 299, 299))  # B, C, H, W

    metric = ISScore(splits=5, is_gpu=torch.cuda.is_available())

    mean_score, std_score = metric.calculate_is(fake_images)

    print(f"Test IS Score: {mean_score:.4f} ± {std_score:.4f}")

if __name__ == "__main__":
    test_inception_score()
