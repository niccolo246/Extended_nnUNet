# Modified nnUNet - Extending the nnUNet Architecture Component.

nnUNet was developed by Isensee et al. and further information on the original framework may be found by reading the following paper:


    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
    for deep learning-based biomedical image segmentation. Nature Methods, 1-9.
    
The nnUNet is a fully automated and generalisable framework which automatically configures the full training pipeline for any medical segmentation task it is applied on, taking into account dataset properties and hardware constraints.  

The nnUNet utilises a standard UNet type architecture which is self-configuring in terms of both depth and hyperparameters. 
We provide code which extends the original nnUNet so as to allow the use of more advanced UNet variations which involve the integration of residual blocks, dense blocks, and inception blocks. 

Users can then easily experiment with a range of different UNet architectural variations which may differ in performance depending on the dataset in question. This is evidenced by the follwing paper:


    Paper coming soon ...

Note: Our code adapts and modifies the original nnUNet code developed by Isensee et al. which can be found at: https://github.com/MIC-DKFZ/nnUNet


# Usage

Below a brief guide to using the modified nnUNet framework is presented which is based on the original nnUNet guide; however, for a more detailed/insightful explanation please refer to the original nnUNet github page mentioned earlier (https://github.com/MIC-DKFZ/nnUNet).

### Installation

To install clone the git page and use pip install. Make sure latest version of PyTorch is installed. 


          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
        
### A. Experiment Planning and Preprocessing

Ensure data is in correct format compatible with nnUNet - refer to [original nnUNet page](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for details. Furthermore paths and relevant folders must be correctly set up as shown [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

To commence experiment planning perform following steps (XXX is the respective task ID):

##### 1) Run basic planning: 

```bash
nnUNet_plan_and_preprocess -t XXX 
```

##### 2) Run planning for custom model: 

##### Residual UNet:

```bash
nnUNet_plan_and_preprocess -t XXX -p nnUNetPlans_ResidualUNet_v2.1 -tr nnUNetTrainerV2_ResidualUNet
```

##### Inception UNet:

```bash
nnUNet_plan_and_preprocess -t XXX -p nnUNetPlans_InceptionUNet_v2.1 -tr nnUNetTrainerV2_InceptionUNet
```

##### Dense UNet:

```bash
nnUNet_plan_and_preprocess -t XXX -p nnUNetPlans_DenseUNet_v2.1 -tr nnUNetTrainerV2_DenseUNet
```

### B. Network Training

We here concentrate on training demonstrations using the 3D full-resolution configuration for the UNet architecture variant. 

Run the following depening on which architecture one wishes to experiemnt with:

##### Residual UNet:

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2_ResidualUNet TASK_NAME_OR_ID FOLD -p nnUNetPlans_ResidualUNet_v2.1
```

##### Inception UNet:

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2_InceptionUNet TASK_NAME_OR_ID FOLD -p nnUNetPlans_InceptionUNet_v2.1
```

##### Dense UNet:

```bash
nnUNet_train 3d_fullres nnUNetTrainerV2_DenseUNet TASK_NAME_OR_ID FOLD -p nnUNetPlans_DenseUNet_v2.1
```







