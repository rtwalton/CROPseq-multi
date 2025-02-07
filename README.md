<h1 align="center"> CROPseq-multi: a versatile solution for multiplexed perturbation and decoding in pooled CRISPR screens</h1>

... now even more versatile(!) with CROPseq-multi-v2

### Code for library design and NGS analysis of CROPseq-multi libraries.

<p align="center">
<img src="https://github.com/rtwalton/CROPseq-multi/blob/CSMv2/input_files/CSMv2_cartoon.png" alt="CROPseq-multi illustration" width="500"/>

### Installation:
#### (Mac OS)

Download the repository, navigate to the `CROPseq-multi` directory, and run the following command to create the environment `CSM`.
```
conda env create -f CSM_env.yml
```
Activate the environment:
```
conda activate CSM
```
#### (systems other than Mac OS)

Create a Python 3.9 environment with the following common python packages:
```
biopython
glob
gzip
itertools
jupyter-lab
matplotlib
numpy
os
pandas
random
seaborn
sys
time
tqdm
```

### Usage:

Jupyter notebooks provide example workflows. 

From the `CROPseq-multi` directory and `CSM` environment, run:

```
jupyter-lab
```
and select a jupyter notebook from the `notebooks` folder
