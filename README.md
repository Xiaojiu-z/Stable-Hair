# Official Repo for Stable-Hair
<a href='https://xiaojiu-z.github.io/Stable-Hair.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/pdf/2407.14078'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 

**Stable-Hair: Real-World Hair Transfer via Diffusion Model**

*Yuxuan Zhang, Qing Zhang, Yiren Song, Jiaming Liu*

<img src='assets/teaser_.jpg'>

## Abstract
Current hair transfer methods struggle to handle diverse and intricate hairstyles, thus limiting their applicability in real-world scenarios. In this paper, we propose a novel diffusion-based hair transfer framework, named Stable-Hair, which robustly transfers a wide range of real-world hairstyles onto user-provided faces for virtual hair try-ons. To achieve this goal, our Stable-Hair framework is designed as a two-stage pipeline while making minimal changes to the original diffusion structure. In the first stage, we train a Bald Converter alongside stable diffusion to remove hair from the user-provided face images, resulting in bald images. In the second stage, we specifically designed three modules: a Hair Extractor, a Latent IdentityNet, and Hair Cross-Attention Layers to transfer the target hairstyle with highly detailed and high-fidelity to the bald image. Specifically, the Hair Extractor is trained to encode reference images with the desired hairstyles. To preserve the consistency of identity content and background between the source images and the transfer results, we employ a Latent IdentityNet to encode the source images. With the assistance of our Hair Cross-Attention Layers in the U-Net, we can accurately and precisely transfer the highly detailed and high-fidelity hairstyle to the bald image. Extensive experiments have demonstrated that our approach delivers state-of-the-art results among existing hair transfer methods.
<img src='assets/method.jpg'>

## Todo List
1. - [ ] Stage1 (Standardize to Bald Stage) inference code 
2. - [ ] Stage1 pre-trained weights 
3. - [ ] Stage2 (Hair Transfer Stage) inference code
4. - [ ] Stage2 pre-trained weights
5. - [ ] All the Training code

## Demos

Coming soon.

## Code

Coming soon.

## Cite
```
@misc{zhang2024stablehairrealworldhairtransfer,
      title={Stable-Hair: Real-World Hair Transfer via Diffusion Model}, 
      author={Yuxuan Zhang and Qing Zhang and Yiren Song and Jiaming Liu},
      year={2024},
      eprint={2407.14078},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14078}, 
}
```
