# A novel Transfer Learning-Based Adaptive Weight Physics-Informed Neural Network for Groundwater Modeling in the middle reaches of Heihe River Basin, northwestern China

In this paper, we propose a novel framework, the Transfer Learning-based Adaptively Weighted Physics-Informed Neural Network (TL-AWPINN), to simulate groundwater in the middle reaches of the Heihe River Basin. The core of this model is the integration of the governing differential equation of groundwater flow as prior knowledge and the design of a physics-informed loss function with adaptive weights. This adaptive weighting mechanism dynamically balances different physical constraints, thereby enhancing the model's physical consistency and predictive accuracy. 

## Data

This study employs two distinct types of datasets for verification experiments. First, as a synthetic case, one synthetic scenario S featuring 2D heterogeneous unconfined aquifers is constructed. Second, the middle reaches of the Heihe River Basin serve as the real-world study area, where a three-dimensional groundwater numerical model covering the period from 1986 to 2008 with a spatial resolution of 1 km×1 km is developed.The Modflow simulation data of the Heihe River Basin can be found at https://github.com/ccfs-cup/Adaptive-Weight-Physics-Informed-Neural-Network/tree/master/GWPINN/dataset. The fine-tuned data can be found in train_output_data.xlsx in the https://github.com/ccfs-cup/Adaptive-Weight-Physics-Informed-Neural-Network/tree/master/GWPINN directory.

## Requirements

GWPGNN works in Python3.9.13 In order to use the GWPGNN successfully, the following site-packages are required:
tensorflow 2.10.0
numpy 1.23.5
pandas 1.5.3
matplotlib 3.7.1
pyDOE 0.3.8
pyparsing 3.0.9
The latest GWPGNN can work in linux-Ubuntu 20.04.6

## Quick verification

The rapid test file for prediction results is used to save the prediction results and fine-tuning results of the trained model, so as to visualize the prediction curves of the Da Man, No. 13 Well and Zhangye Farm.

## Folder Description

1. SA-PINNs-master is the reference code for the adaptive weight optimization method proposed in the paper. To understand the adaptive weight optimization specifically, focus on the sections where the weights parameter is configured and applied.

2. UNPINN corresponds to the code for simulating two-dimensional heterogeneous unconfined aquifers. The simulated area S used in the experiment corresponds to the S1 area in the code.

3. GWPINN corresponds to the training code for the middle reaches of the Heihe River Basin. You can set a certain proportion of PDE sampling points so that the memory usage is manageable for the server.



