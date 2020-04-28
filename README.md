# Graph2Taxo

Graph2Taxo is a GNN-based cross-domain transfer framework for taxonomy construction. It uses a noisy graph constructed from automatically extracted noisy hyponym-hypernym candidate pairs, and a set of taxonomies for some known domains for training. The learned model is then used to generate taxonomy for a new unknown domain given a set of terms for that domain.
 
If you use this system, please cite the following paper -
```
@inproceedings{chao2020-g2t,
    title={Taxonomy Construction of Unseen Domains via Graph-based Cross-Domain Knowledge Transfer},
    author={Chao Shang and Sarthak Dash and Md Faisal Mahbub Chowdhury and Nandana Mihindukulasooriya and Alfio Gliozzo},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020) },
    publisher = "Association for Computational Linguistics",
    year      = {2020},
}
```


## Installation

Install PyTorch from the [official website](https://pytorch.org/get-started/) or using Anaconda.

## Initializing Git submodules. 

After cloning the repo, if you need to process the data, please use the command `git submodule update` to initialize the dependent submodules. This will clone [TaxoRL](https://github.com/morningmoni/TaxoRL) and [TAXI](https://github.com/uhh-lt/taxi/) projects that are used to reproduce data from existing experiments. 

    git submodule update

## Train model

### Dataset

TAXI data is given in the "data/TAXI_dataset" folder. Data from TaxoRL paper is given in the "data/TaxoRL_dataset" folder.

When you process the data, you can run:

    python preprocess.py

### Train model

When you train the model, you can run:

    python train.py


## Design your own model

You can directly modify the GRAPH2TAXO model in the "models.py" file.

## Acknowledgments
[GCN](https://github.com/tkipf/gcn), [TaxoRL](https://github.com/morningmoni/TaxoRL), [TAXI](https://github.com/uhh-lt/taxi) and [SACN](https://github.com/JD-AI-Research-Silicon-Valley/SACN).



