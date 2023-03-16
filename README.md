
<br />
<p align="center">

  <h1 align="center"><a href="https://arxiv.org/pdf/2103.09671.pdf">Fourier Transform of Percoll Gradients Boosts CNN Classification of Hereditary Hemolytic Anemias</a></h1>

  <a href="https://arxiv.org/pdf/2103.09671.pdf">
    <img src="https://lmoyasans.github.io/images/percoll.png" alt="Logo" width="100%">
  </a>

  <p align="center">
     2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)
    <br />
    <a href="https://scholar.google.es/citations?hl=es&user=9KVOSLkAAAAJ"><strong>Ario Sadafi</strong></a>
    ·
    <a href="https://scholar.google.es/citations?hl=es&user=u-6GyqwAAAAJ"><strong>Lucia Moya-Sans</strong></a>
    ·
    <a href="https://scholar.google.es/citations?hl=es&user=qfMSEfsAAAAJ"><strong>Asya Makhro</strong></a>
    ·
    <a href="https://scholar.google.es/citations?hl=es&user=Y4I1DxcAAAAJ"><strong>Leonid Livshits</strong></a>
    ·
    <a href="https://scholar.google.es/citations?user=kzoVUPYAAAAJ&hl=es"><strong>Nassir Navab</strong></a>
    ·
    <a href="https://deepai.org/profile/anna-bogdanova"><strong>Anna Bogdanova</strong></a>
    ·
    <a href="https://scholar.google.es/citations?user=CPuApzoAAAAJ&hl=es&oi=ao"><strong>Shadi Albarqouni</strong></a>
    ·
    <a href="https://scholar.google.es/citations?user=Wg9zjqEAAAAJ&hl=es"><strong>Carsten Marr</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/pdf/2103.09671.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
  </p>
</p>

<br />
<br />



## Overview

This repository contains the source code of the paper **FOURIER TRANSFORM OF PERCOLL GRADIENTS BOOSTS CNN CLASSIFICATION OF HEREDITARY HEMOLYTIC ANEMIAS** by *Ario Sadafi* *, *Lucia Moya-Sans* *, *Asya Makhro*, *Leonid Livshits*, *Nassir Navab*, *Anna Bogdanova*, *Shadi Albarqouni*, and *Carsten Marr* from 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI).

### Abstract

Hereditary hemolytic anemias are genetic disorders that affect the shape and density of red blood cells. Genetic tests currently used to diagnose such anemias are expensive and unavailable in the majority of clinical labs. Here, we propose a method for identifying hereditary hemolytic anemias based on a standard biochemistry method, called Percoll gradient, obtained by centrifuging a patient's blood. Our hybrid approach consists on using spatial data-driven features, extracted with a convolutional neural network and spectral handcrafted features obtained from fast Fourier transform. We compare late and early feature fusion with AlexNet and VGG16 architectures. AlexNet with late fusion of spectral features performs better compared to other approaches. We achieved an average F1-score of 88% on different classes suggesting the possibility of diagnosing of hereditary hemolytic anemias from Percoll gradients. Finally, we utilize Grad-CAM to explore the spatial features used for classification.

### Citation

If you are using this code in academic research, we would be grateful if you cited our paper:

```bibtex
@INPROCEEDINGS{9433788,
  author={Sadafi, Ario and Sans, Lucía María Moya and Makhro, Asya and Livshits, Leonid and Navab, Nassir and Bogdanova, Anna and Albarqouni, Shadi and Marr, Carsten},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)}, 
  title={Fourier Transform of Percoll Gradients Boosts CNN Classification of Hereditary Hemolytic Anemias}, 
  year={2021},
  volume={},
  number={},
  pages={966-970},
  doi={10.1109/ISBI48211.2021.9433788}}
}
```
