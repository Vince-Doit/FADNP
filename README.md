# FADNP
Fast Discriminant Analysis With Adaptive Reconstruction Structure Preserving

This repository contains the code for the paper titled "Fast Discriminant Analysis With Adaptive Reconstruction Structure Preserving," published in TNNLS in 2024. The paper addresses several key challenges in the field of discriminant analysis. First, traditional methods learn reconstruction coefficients based on the collaborative representation of all sample pairs, which results in a training time that scales cubically with the number of samples, making the process computationally expensive and impractical for large datasets. Second, these coefficients are typically learned in the original feature space, ignoring the interference of noise and redundant features, which can obscure the intrinsic structure of the data and lead to suboptimal performance. Third, there is often a reconstruction relationship between heterogeneous samples, which can artificially inflate the similarity between dissimilar samples in the subspace, degrading the discriminative power of the model. To address these challenges, the paper proposes a novel approach that adapts the reconstruction structure to preserve the essential characteristics of the data while mitigating the impact of noise and redundancy, thereby improving both the efficiency and effectiveness of discriminant analysis.

The repo also hosts some baseline systems as we compared in the paper. We would like to thank the authors of the baseline systems for their codes. If any baseline systems cannot be licensed freely here, please drop me an email, so we can remove it from the collection.

If you find this repo useful, please kindly cite the paper below.

@article{zhao2024fast,
  title={Fast Discriminant Analysis With Adaptive Reconstruction Structure Preserving},
  author={Zhao, Xiaowei and Nie, Feiping and Wang, Rong and Li, Xuelong},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={35},
  number={8},
  pages={11106--11115},
  year={2024},
  publisher={IEEE}
}
