# GSDN-F and GSDN-EF
This repository provides a reference implementation of *GSDN-F* and *GSDN-EF* as described in the [paper](https://arxiv.org/abs/2006.04386) "**Understanding Graph Neural Networks from Graph Signal Denoising Perspectives**". 

### Requirements
Install the following packages:

- [pytorch](https://pytorch.org/get-started/locally/)
- [torch_geometric](https://github.com/rusty1s/pytorch_geometric)
- scikit-learn
- networkx

### Basic Usage
```
$ python gsdnf.py --input cora --alpha 0.6 --K 4 --epochs 200 --lr 0.02 --hidden_num 16
$ python gsdnef.py --input cora --alpha 0.6 --K 4 --epochs 200 --lr 0.02 --hidden_num 16
```
>noted: your can just checkout *gsdnf.py* and *gsdnef.py* to get what you want.

### Citing
If you find *GSDN-F* and/or *GSDN-EF* useful in your research, please cite our paper:

	@misc{2006.04386,
	 Author = {Guoji Fu and Yifan Hou and Jian Zhang and Kaili Ma and Barakeel Fanseu Kamhoua and James Cheng},
	 Title = {Understanding Graph Neural Networks from Graph Signal Denoising Perspectives},
	 Year = {2020}
	}
