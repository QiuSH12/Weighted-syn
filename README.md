# Weighted-MRI synthesis from MR Mulitasking spatial factors
This repository includes the source code (deep learning part) for the paper: _**Direct Synthesis of Multi-Contrast Brain MR Images from MR Multitasking Spatial Factors Using Deep Learning**_.

In this work, a deep learning approach was developed to synthesize conventional contrast-weighted images in the brain from MR Multitasking spatial factors.

**MR Multitasking** is a representative approach with the capability to acquire multi-parametric maps (e.g., T1, T2, T1rho, ...) in a single scan, which has great potential to provide diverse information in a short acquisition time. With the proposed deep learning method, both quantitative parameter maps and multiple-contrast weighted images can be obtained from a single MR Multitasking scan, with no additional time cost for weighted MRI acquisition.

The network used in this work is based on a 2D U-net architecture:

![image](https://user-images.githubusercontent.com/40025501/232633616-12c6ed0e-db37-4d52-936b-0c6534013844.png)

The input data is the standardized spatial factor (5-channel), and the target images include T1 MPRAGE, T1 GRE, and T2 FLAIR. The loss function is a combination of L1 loss and SSIM loss.

The implementation is based on TensorFlow 2.6.0 and Python 3.8.0.

## Data
The training and evaluation data is available on Box: https://cedars.box.com/s/hai78q3t3ndl3dikejyebw9rqg7b25tg.
Eighteen subjects (fifteen diagnosed or suspected MS patients, three healthy volunteers) are included in the dataset. All data is preprocessed and deidentified. For each subject, the following files are available:
* U0_5ch_Rig.nii: the standardized spatial factor from MR Multitasking (5-channel input data)
* T1mprage_Rig.nii, T1gre_Rig.nii, T2flair_RIg.nii: weighted images (labels)
* mask_auto.nii: a automatically generated mask, to remove background signal and the slices around the edge of imaging volume with low signal intensity
* T1mprage_rig_brain_mask.nii.gz, T1mprage_rig_brain_seg.nii.gz: skull-stripped mask (generated by FSL-BET) and GM/WM/CSF segmentation (generated by FSL-FAST)

## Further reading
**About MR Multitasking:**
* Christodoulou AG, Shaw JL, Nguyen C, et al. Magnetic resonance multitasking for motion-resolved quantitative cardiovascular imaging. _Nat Biomed Eng_. 2018;2(4):215-226. https://doi.org/10.1038/s41551-018-0217-y
* Ma S, Wang N, Fan Z, et al. Three-dimensional whole-brain simultaneous T1, T2, and T1rho quantification using MR Multitasking: Method and initial clinical experience in tissue characterization of multiple sclerosis. _Magn Reson Med_. 2021;85(4):1938-1952. https://doi.org/10.1002/mrm.28553

**Previous presentation in ISMRM 2021:**
* Qiu S, Chen Y, Ma S, et al. Direct Synthesis of Multi-Contrast Images from MR Multitasking Spatial Factors Using Deep Learning. In Proc ISMRM 2021. P. 2429. https://archive.ismrm.org/2021/2429.html

**A previous work estimating T1 and T2 maps from conventional weighted images using the same DL architecture:**
* Qiu S, Chen Y, Ma S, et al. Multiparametric mapping in the brain from conventional contrast-weighted images using deep learning. _Magn Reson Med_. 2021.  https://doi.org/10.1002/mrm.28962

## Contact info
Shihan Qiu, qiushihan19@gmail.com
