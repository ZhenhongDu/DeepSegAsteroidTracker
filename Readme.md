 [![Project](https://img.shields.io/badge/project-DSAT%20-blue.svg)](https://zhenhongdu.github.io/asteroid_tracking/)  [![Github commit](https://img.shields.io/github/last-commit/zhenhongdu/DeepSegAsteroidTracker)](https://github.com/zhenhongdu/DeepSegAsteroidTracker/)  [![Github stars](https://img.shields.io/github/stars/zhenhongdu/DeepSegAsteroidTracker?color=hex)](https://github.com/zhenhongdu/DeepSegAsteroidTracker/)  ![Program langurage](https://img.shields.io/badge/Python-3.8-orange)  <img src="https://badges.toozhao.com/badges/01HJ6B2Z74GM371X2NKHSJ6MF9/green.svg" /> 



<p align="center">
<h1 align="center">DSAT: <strong>D</strong>eep <strong>S</strong>egmentation-assisted <strong>A</strong>steroid <strong>T</strong>racking</h1> </h1>
</p>


Note: We have released  the code. We will update the readme file as soon as possible!

## Contents



<img src="images/logo.jpg" width="170" align="right">

- [Overview](#overview)
- [Installation](#installation)
- [Demo](#demo)
- [Data](#results)
- [Acknowledge](#acknowledge)
- [Citation](#citation)

# Overview



<img src="images/tracking_result.gif" width="360" align="left">

DSAT aims to find faint moving asteroids using the digital imaging process method. In short, DSAT first extracts all potential faint objects with the help of deep learning-based segmentation. After that, a multi-frame tracking algorithm was developed to find real asteroids from the segmentation results. We utilize a distance tolerance criterion to help the failure detection of asteroids in complex situations. More detailed information can be found in our paper. DAST was developed using Python. 

<br>

# Installation



We recommend using Anaconda or Miniconda to manage the package environment. You can clone this repository, then go to the folder and install the dependencies using Anaconda.

- Try a demo (w/o pytorch-gpu) --> for simulation or asteroid tracking

```python
conda env create -f environment.yml
```

- Full dependencies (w pytorch-gpu)--> for simulation, asteroid tracking, network training and inference

```
conda env create -f environment_full.yml
```

After installation, you can activate the created environment with 'conda activate DSAT'.


# Demo






# Data

The simulated data will be uploaded to zenodo after it is orgranized.




# Acknowledge






# Citation


