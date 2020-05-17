# Structure-constrained CycleGAN

Tenserflow implementation for our conference paper "Unpaired Brain MR-to-CT Synthesis Using a Structure-Constrained CycleGAN".

## Basic information:
**Input**: unpaired CT and MR volumes. <br>
**Output**: the synthetic MR/CT volume corresponding to input CT/MR volume.


## Train & Test:
1. There are many settings in main.py that needs to be changed according to your applications. Besides, it contains three modes, respectively train, validation and test. <br>
2. The experiments are performed on unpaired MR and CT data, including the training, validation and testing subjects.


## Citation:
If you use this code for your research, please cite our paper:
> @inproceedings{yang2018, 
> <br> title={Unpaired brain MR-to-CT synthesis using a structure-constrained CycleGAN}, 
> <br> author={Yang, Heran and Sun, Jian and Carass, Aaron and Zhao, Can and Lee, Junghoon and Xu, Zongben and Prince, Jerry},
> <br> booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},
> <br> pages={174--182},
> <br> year={2018}
> <br>}


## Reference
The tensorflow implementation of CycleGAN: https://github.com/XHUJOY/CycleGAN-tensorflow
