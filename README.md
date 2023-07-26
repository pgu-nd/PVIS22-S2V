# Scalar2Vec: Translating Scalar Fields to Vector Fields via Deep Learning
This repository contains the PyTorch implementation for paper "Scalar2Vec: Translating Scalar Fields to Vector Fields via Deep Learning".

# Prerequisites
* Linux
* CUDA >= 10.0
* Python >= 3.7
* Numpy
* Pytorch >= 1.0

# How to run the code
* First, change the directory path and parameter settings (e.g., batch size, samples etc.) in main.py. 
* Second, to train the model, simply call python3 main.py --train train. 
* Third, to inference the new data, simply call python3 main.py --train infer.

# Citation
@inproceedings{gu2022scalar2vec,<br/>
  title={Scalar2Vec: Translating Scalar Fields to Vector Fields via Deep Learning},<br/>
  author={Gu, Pengfei and Han, Jun and Chen, Danny Z and Wang, Chaoli},<br/>
  booktitle={2022 IEEE 15th Pacific Visualization Symposium (PacificVis)},<br/>
  pages={31--40},<br/>
  year={2022},<br/>
  organization={IEEE}<br/>
}

# Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1455886, CNS-1629914, DUE-1833129, IIS-1955395, IIS-2101696, and OAC-2104158.
