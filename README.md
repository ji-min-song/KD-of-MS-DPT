# KD of MS-DPT for Self-supervised Depth Estimation
Knowledge Distillation of Multi-scale Dense Prediction Transformer for Self-supervised Depth Estimation

## Preparation
1. Prepare the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/) following below code.
~~~   
$ python test.py arguments_test_eigen.txt   
~~~   
The above code and instruction are provided by [BTS](https://github.com/cleinc/bts)
2. Download [DPT large model's initial weight](https://drive.google.com/file/d/1TWpC6cRCpPXLIAd20gz_i31z65CnV2Ah/view?usp=share_link) pretrained on the MIX 6 dataset.  
The above parameters are provided by [DPT](https://github.com/isl-org/DPT)
3. Download [Ours student's pretrained weight](https://drive.google.com/file/d/1PB6oZiEZzYR7qvRHIKogub1jezpoBbTv/view?usp=share_link) trained on the KITTI dataset.  
4. Finally, place the two downloaded weights in the './models/DPT_student'   
~~~   
──┬ KD-of-MS-DPT
  ├── train_test_inputs
  ├──┬ models
     ├── Monodepth2
     ├── DPT_teacher
     ├──┬ DPT_student
        ├── dpt_large-midas-2f21e586.pt ★
        ├── student_depth ★
                 :
                 :
                 :
~~~   

## Train

## Inference
~~~   
$ python test.py arguments_test_eigen.txt   
~~~   

## Acknowledgement
This repository makes extensive use of code from the [BTS](https://github.com/cleinc/bts) & [DPT](https://github.com/isl-org/DPT) Github repository.  
We thank the authors for open sourcing their implementation.
