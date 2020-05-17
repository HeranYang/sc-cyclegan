# Structure-constrained cycleGAN
# Input: unpaired CT and MR volumes.
# Output: the synthetic MR/CT volume corresponding to input CT/MR volume.
heran 2018/09/18

# Note:
## 1. There are many settings in main.py that needs to be changed according to your applications. Besides, it contains three modes, respectively train, validation and test.
## 2. The experiments are performed on Junghoon's data, where 27 subjects are used for training, 3 subjects for validation and 15 for test.


# Reference: Yang, H., Sun, J., Carass, A., Zhao, C., Lee, J., Xu, Z., & Prince, J. L. Unpaired Brain MR-to-CT Synthesis using a Structure-Constrained CycleGAN. In MICCAI 2018 Workshop: Deep Learning in Medical Image Analysis (DLMIA).