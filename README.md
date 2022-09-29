# DPSSCL
Anonymous repo for DP-SSCL submission to ICLR 2023

DP-SSCL is a generalized framework to augment supervised continual learning algorithms for semi-supervised learning via data programming. Since components of semi-supervised learning (i.e. pseudo-labeling and continual learning) are modularized, this framework allows to use any continual learners in plug-and-play style.

## Pre-requisite
This code for experiment requires pytorch==1.6.0 and tensorflow==1.14.
We recommend to use conda to install required python packages.

## How to run code
There are two main code: **main.py** (DP-SSCL without WLFs transfer) and **main_wlfs_transfer.py** (DP-SSCL with WLFs transfer).
### Prepare data files
1. Run **generate_binary_task_data.py**
2. Run **generate_task_data.py**
### Experiments
1. If running DP-SSCL without transferring WLFs, run **main.py** as following commands:

   (MNIST) ```python3 main.py --gpu 0 --data_type MNIST_mako --data_unlabel 120 --model_type Hybrid_DFCNN --test_type 4 --lifelong --num_clayers 2 --mako_baseline snorkel --confusion_matrix --save_mat_name dpsscel_mnist_dfcnn.mat```

   (CIFAR-10) ```python3 main.py --gpu 0 --data_type CIFAR10_mako --data_unlabel 400 --model_type Hybrid_DFCNN --test_type 4 --lifelong --num_clayers 4 --mako_baseline snorkel --confusion_matrix --save_mat_name dpsscel_cifar10_dfcnn.mat```
   
   Here, you can change the size of unlabeled data after data_unlabel and the continual learner after model_type (Hybrid_HPS, Hybrid_TF, Hybrid_DFCNN). Of course, feel free to specify the result file name after save_mat_name.
2. If running DP-SSCL with transferring WLFs, run **main_wlfs_transfer.py** as following commands:

   (MNIST) ```python3 main_wlfs_transfer.py --gpu 0 --data_type MNIST_mako --data_label 100 --data_unlabel 100 --model_type Hybrid_DFCNN --test_type 4 --lifelong --num_clayers 2 --mako_transfer_score_threshold 0.7 --mako_transfer_knowledge_prob 0.6 --mako_baseline snorkel --confusion_matrix --weak_labelers_min 10 --weak_labelers_max 25 --weak_labelers_keep 10 --finetune_wls --save_mat_name dpsscel_withTransfer_mnist_dfcnn.mat```
   
   (CIFAR-10) ```python3 main_wlfs_transfer.py --gpu 0 --data_type CIFAR10_mako --data_label 200 --data_unlabel 200 --model_type Hybrid_DFCNN --test_type 4 --lifelong --num_clayers 4 --mako_transfer_score_threshold 0.7 --mako_transfer_knowledge_prob 0.6 --mako_baseline snorkel --confusion_matrix --weak_labelers_min 10 --weak_labelers_max 25 --weak_labelers_keep 10 --finetune_wls --save_mat_name dpsscel_withTransfer_cifar10_dfcnn.mat```
   
   Here, you can change the size of labeled and unlabeled data after data_label and data_unlabel, the continual learner after model_type, and the result file name after save_mat_name.