# KD of MS-DPT for Self-supervised Depth Estimation
Knowledge Distillation of Multi-scale Dense Prediction Transformer for Self-supervised Depth Estimation   
abstract & demo video

## Demo video
<p align="center">
  <img src="assets/input.gif" width="600" />
  <img src="assets/monodepth2.gif" width="600" />
  <img src="assets/teacher.gif" width="600" />
  <img src="assets/student.gif" width="600" />
</p>

## Environment setup   
~~~   
$ conda create -n MS-DPT python=3.9
$ conda activate MS-DPT

$ conda install -c anaconda cudatoolkit=11.3
$ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

$ conda install -c conda-forge opencv
$ conda install -c conda-forge einops 
$ conda install -c conda-forge tensorboardx 
$ conda install -c conda-forge tqdm 
$ conda install -c conda-forge timm 
$ conda install -c conda-forge matplotlib 
~~~  

## Data preparation   
### KITTI dataset
Prepare the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/) following below code.   
   
Download and unzip the [improved ground truth depthmaps](https://www.cvlibs.net/download.php?file=data_depth_annotated.zip).   
And you can download original ground truth depthmaps named "projected raw LiDaR scans data set (5 GB)" in [here](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

~~~   
$ cd ~/workspace/dataset
$ mkdir kitti_dataset && cd kitti_dataset
$ mv ~/Downloads/data_depth_annotated.zip .
$ unzip data_depth_annotated.zip  
~~~   

Download and unzip KITTI raw dataset.

~~~   
$ cd ~/workspace/dataset/kitti_dataset
$ aria2c -x 16 -i ../../KD-of-MS-DPT/kitti_archives_to_download.txt
$ parallel unzip ::: *.zip  
~~~    

The above codes and instructions are provided by [BTS](https://github.com/cleinc/bts).   
   
And we duplicate the KITTI dataset and compress for teacher network following [Monodepth2](https://github.com/nianticlabs/monodepth2) which is the baseline of teacher network.   

~~~   
$ cp ~/workspace/dataset/kitti_dataset/ ~/workspace/dataset/kitti_raw_dataset/
$ find ~/workspace/dataset/kitti_dataset/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
~~~   
   
### Pretrained models
Download [the weight of the backbone of the DPT large model](https://drive.google.com/file/d/1TWpC6cRCpPXLIAd20gz_i31z65CnV2Ah/view?usp=share_link) pretrained on the MIX 6 dataset. This weight are provided by [DPT](https://github.com/isl-org/DPT)   
Download [the pretrained weight of our student network](https://drive.google.com/file/d/1PB6oZiEZzYR7qvRHIKogub1jezpoBbTv/view?usp=share_link) trained on the KITTI dataset.   
Place the two downloaded weights in the './models/DPT_student' like below folder structure.   
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
Our pipeline needs a total of 2 gpu, one for teacher network and the other for student network.    
You can choose gpus by specifying it in the 'arguments_train_eigen.txt' file
~~~   
$ python train.py arguments_train_eigen.txt   
~~~ 

## Inference
For inference, one gpu is needed because only the student network is required.    
~~~   
$ python test.py arguments_test_eigen.txt   
~~~   

## Acknowledgement
This repository makes extensive use of code from the [BTS](https://github.com/cleinc/bts) & [DPT](https://github.com/isl-org/DPT) Github repository.  
We thank the authors for open sourcing their implementation.   
   
<!--We are grateful for the open-source implementation of [BTS](https://github.com/cleinc/bts) & [DPT](https://github.com/isl-org/DPT) Github repository available on GitHub.-->

