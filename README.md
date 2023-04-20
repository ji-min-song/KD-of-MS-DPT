# KD of MS-DPT for Self-supervised Depth Estimation
Knowledge Distillation of Multi-scale Dense Prediction Transformer for Self-supervised Depth Estimation

## Preparation
1. Download [DPT large model's initial weight](https://drive.google.com/file/d/1TWpC6cRCpPXLIAd20gz_i31z65CnV2Ah/view?usp=share_link) pretrained on the MIX 6 dataset.  
The above parameters are provided by [DPT](https://github.com/isl-org/DPT)
2. Download [Ours student's pretrained weight](https://drive.google.com/file/d/1PB6oZiEZzYR7qvRHIKogub1jezpoBbTv/view?usp=share_link) trained on the KITTI dataset.  
3. Finally, place the two downloaded weights in the './models/DPT_student'   
~~~   
-- KD-of-MS-DPT
   -- train_test_inputs
   -- models
      -- Monodepth2
      -- DPT_student
      -- DPT_student
~~~   



## Train

## Inference
~~~   
python test.py arguments_test_eigen.txt   
~~~   

## Acknowledgement
This repository makes extensive use of code from the [BTS](https://github.com/cleinc/bts) & [DPT](https://github.com/isl-org/DPT) Github repository.  
We thank the authors for open sourcing their implementation.
