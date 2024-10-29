# Official Repo for Stable-Hair
<a href='https://xiaojiu-z.github.io/Stable-Hair.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/pdf/2407.14078'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 

**Stable-Hair: Real-World Hair Transfer via Diffusion Model**

*Yuxuan Zhang, Qing Zhang, Yiren Song, Jiaming Liu*

<img src='assets/teaser_.jpg'>

## Abstract
Current hair transfer methods struggle to handle diverse and intricate hairstyles, limiting their applicability in real-world scenarios. In this paper, we propose a novel diffusion-based hair transfer framework, named Stable-Hair, which robustly transfers a wide range of real-world hairstyles to user-provided faces for virtual hair try-on. To achieve this goal, our Stable-Hair framework is designed as a two-stage pipeline. In the first stage, we train a Bald Converter alongside stable diffusion to remove hair from the user-provided face images, resulting in bald images. In the second stage, we specifically designed a Hair Extractor and a Latent IdentityNet to transfer the target hairstyle with highly detailed and high-fidelity to the bald image. The Hair Extractor is trained to encode reference images with the desired hairstyles, while the Latent IdentityNet ensures consistency in identity and background. To minimize color deviations between source images and transfer results, we introduce a novel Latent ControlNet architecture, which functions as both the Bald Converter and Latent IdentityNet. After training on our curated triplet dataset, our method accurately transfers highly detailed and high-fidelity hairstyles to the source images. Extensive experiments demonstrate that our approach achieves state-of-the-art performance compared to existing hair transfer methods.
<img src='assets/method.jpg'>

## Todo List
1. - [x] Stage1 (Standardize to Bald Stage) inference code 
2. - [x] Stage1 pre-trained weights 
3. - [x] Stage2 (Hair Transfer Stage) inference code
4. - [x] Stage2 pre-trained weights
5. - [x] Training code

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
