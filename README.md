# Introduction

**commonroad-geometric (crgeo)** is a Python framework that facilitates deep-learning based research projects in the autonomous driving domain, e.g. related to behavior planning and state representation learning.

At its core, it provides a standardized interface for heterogeneous graph representations of traffic scenes using the [PyTorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) framework.

The package aims to serve as a flexible framework that, without putting restrictions on potential research directions, minimizes the time spent on implementing boilerplate code. Through its object-oriented design with highly flexible and extendable class interfaces, it is meant to be imported via **pip install** and utilized in a plug-and-play manner.
<!--
---

|<img src="img/sumo_sim_temporal_1.gif" width="420" height="330" />|<img src="img/DEU_Munich-1_104-step-0-to-step-400.gif" width="420" height="330"/>|
| ---      | ---       |
|<img src="img/CR-v0-DEU_Munich-1_19-2022-09-21-172426-step-0-to-step-20000.gif" width="420" height="330"/>|<img src="img/occupancy_predictive_training.gif" width="420" height="330"/>|


--- -->

## Highlighted features

- A framework for [PyTorch Geometric-based](https://pytorch-geometric.readthedocs.io/) heterogeneous graph data extraction from traffic scenes and road networks supporting user-made feature computers and edge drawers.
- Built-in functionality for collection and storing of graph-based traffic datasets as [PyTorch datasets](https://pytorch.org/vision/stable/datasets.html).
- Fully customizable live rendering support for showcasing and debugging.
- High-level training infrastructure for crgeo research projects.


<!-- --- -->

<!-- ## High-level package architecture


<img src="img/crgeo_high_level_architecture.svg" width="900" style="margin: 0 auto; overflow: hidden; margin-bottom: 20px" /> -->

---

## Citing this work

If you use this work in your research, please cite it as follows:

```bibtex
@INPROCEEDINGS{meyer2023geometric,
  author={Meyer, Eivind and Brenner, Maurice and Zhang, Bowen and Schickert, Max and Musani, Bilal and Althoff, Matthias},
  booktitle={2023 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Geometric Deep Learning for Autonomous Driving: Unlocking the Power of Graph Neural Networks With CommonRoad-Geometric}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IV55152.2023.10186741}
}
```


# Getting started

The easiest way of getting familiar with the framework is to consult the [tutorial directory](tutorials), which contains a multitude of simple application demos that showcase the intended usage of the package.

### Research guidelines:

- It is highly recommended to incorporate the package's extendable rendering capabilities as an integral part of your development workflow. This allows you to visualize what is going on in your experiment, greatly simplifying debugging efforts.
- If you ever find yourself in a situation where you have to modify the internals of this package while working on your research project, it probably means that commonroad-geometric is not flexible enough - please create a corresponding issue.

---

# Installation

The installation script [`scripts/create-dev-environment.sh`](scripts/create-dev-environment.sh) installs the commonroad-geometric package and all its dependencies into a conda environment:

Execute the script inside the directory which you want to use for your development environment.

Note: make sure that the CUDA versions are compatible with your setup.


### Note: Headless rendering
If you want to export the rendering frames without the animation window popping up, please use the command given below.
``` shell
echo "export PYGLET_HEADLESS=..." >> ~/.bashrc
```
You can replace `.bashrc` with `.zshrc`, if you use `zsh`

---
