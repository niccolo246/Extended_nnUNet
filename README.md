# Modified nnUNet - Extending nnUNet Architecure Component.


The nnUNet is, a fully automated and generalisable framework which automatically configures the full training pipeline for any medical segmentation task it is applied on, taking into account dataset properties and hardware constraints.  

nnUNet was developed by Isensee et al. and further information on the original framework may be found by reading the follwing paper:


    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
    for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

The nnUNet utilises a standard UNet type architecture which is self-configuring in terms of both depth and hyperparameters. 
We procide code which extends the original nnUNet so as to allow the use of more advanced UNet variations which involve the integration of residual blocks, dense blocks, and inception blocks. 

