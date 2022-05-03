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


