# HSRL
This repository provides a reference implementation of *HSRL* as described in the [paper](https://arxiv.org/abs/1902.06684). 

### Requirement
Install the following packages:

- [pytorch](https://pytorch.org/get-started/locally/)
- [torch_geometric](https://github.com/rusty1s/pytorch_geometric/blob/master/README.md)
- scikit-learn
- networkx

### Basic Usage
```
$ python gsdnf.py --input cora --output --alpha 0.6 --K 4 --epochs 200 --lr 0.02 --hidden_num 16
```
>noted: your can just checkout gsdnf.py to get what you want.

### Citing
If you find *HSRL* useful in your research, please cite our paper:

	@article{fu2019learning,
	 title={Learning Topological Representation for Networks via Hierarchical Sampling},
	 author={Fu, Guoji and Hou, Chengbin and Yao, Xin},
	 journal={arXiv preprint arXiv:1902.06684},
	 year={2019}
	} 