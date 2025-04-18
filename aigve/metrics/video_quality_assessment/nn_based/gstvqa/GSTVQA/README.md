# GSTVQA
Code for 'Learning Generalized Spatial-Temporal Deep Feature  Representation for No-Reference Video Quality Assessment'. The code is partially borrowed from  [VSFA](https://github.com/lidq92/VSFA).
![image](https://user-images.githubusercontent.com/75255236/121126057-1fbca280-c85a-11eb-9b6d-2d221a83b263.png)


# Environment
* Python 3.6.7
* Pytorch 1.6.0  Cuda V9.0.176 Cudnn 7.4.1

# Running
* Download the pre-extracted multi-scale VGG features of each dataset from [BaiduYun](https://pan.baidu.com/s/1pyl5Yz4opPdoACnqSWLXsw), Extraction code: `gstv`. Then put the features in the path: "./GSTVQA/TCSVT_Release/GVQA_Release/VGG16_mean_std_features/".

* Train:  
  `python  ./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/main.py --TrainIndex=1  
  (TrainIndex=1：using the CVD2014 datase as source dataset; 2: LIVE-Qua; 3: LIVE-VQC; 4: KoNviD）`

* Test:  
  `python  ./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py --TrainIndex=1  
  （TrainIndex=1：using the CVD2014 datase as source dataset; 2: LIVE-Qua; 3: LIVE-VQC; 4: KoNviD）`  

# Details
* The model trained on each above four datasets has been provided in "./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/"
* The code for VGG-based feature extraction is available at: https://mega.nz/file/LXhnETyD#M6vI5M9QqStFsEXCeiMdJ8BWRrLxvRbkZ1rqQQzoVuc
* In the intra-dataset setting, it should be noted that we use 80% of the data for training and the rest 20% data for testing. We haven't used the 20% data for the best epoch selection to avoid testing data leaky, instead, the last epoch is used for performance validation.
