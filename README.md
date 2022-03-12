# Improving Bayesian Graph Convolutional Networks using Markov Chain Monte Carlo Graph Sampling
This is the source code for the implementation of "Improving Bayesian Graph Convolutional Networks using Markov Chain Monte Carlo Graph Sampling" (A. Komanduri). This repo contains the code for the modified and enhanced Bayesian GCN for the node classification task.



## Usage

### Training and evaluating BGCN-NRWS

The script `run.sh` allows for training and evaluating our model. Simply run the script as follows:

`./run.sh`

A trained BGCN-NRWS model can also be tested by modifying the boolean flag in `main.py`, as follows:

`var = True`


## Acknowledgements

This research was supported by the Arkansas Research Alliance and the NSF DART Grant.

### Data acknowledgements
In our paper, we used the following datasets to evaluate our model under strict semi-supervision:

#### Citation Network Datasets for Node Classification
<details closed>
<summary>Cora Dataset</summary>

[Link to dataset](https://paperswithcode.com/sota/node-classification-on-cora)
</details>

<details closed>
<summary>Citeseer Dataset</summary>

[Link to dataset](https://paperswithcode.com/sota/node-classification-on-citeseer#:~:text=The%20CiteSeer%20dataset%20consists%20of,into%20one%20of%20six%20classes.&text=Each%20publication%20in%20the%20dataset,consists%20of%203703%20unique%20words.)
</details>

<details closed>
<summary>Pubmed Dataset</summary>

[Link to dataset](https://paperswithcode.com/sota/node-classification-on-pubmed)
</details>

## Citation

If you use our code or think our work is relevant to yours, we encourage you to cite this paper:

```bibtex
@INPROCEEDINGS{9679954,
  author={Komanduri, Aneesh and Zhan, Justin},
  booktitle={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Neighborhood Random Walk Graph Sampling for Regularized Bayesian Graph Convolutional Neural Networks}, 
  year={2021},
  volume={},
  number={},
  pages={903-908},
  doi={10.1109/ICMLA52953.2021.00149}}
```
