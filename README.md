# Graph2Taxo

This is a PyTorch implementation of the ACL-2020 paper "Taxonomy Construction of Unseen Domains via Graph-based Cross-Domain Knowledge Transfer".

## Installation

Install PyTorch using official website or Anaconda.

## Train model

### Dataset

TAXI datas is given in the "data/TAXI_dataset" folder. Data from TaxoRL paper is given in the "data/TaxoRL_dataset" folder.

When you process the data, you can run:

    python preprocess.py

### Train the model

When you train the model, you can run:

    python train.py


## Model Structure

You can directly modify the GRAPH2TAXO model in the "models.py" file.

## Acknowledgments
[GCN](https://github.com/tkipf/gcn), [TaxoRL](https://github.com/morningmoni/TaxoRL), [TAXI](https://github.com/uhh-lt/taxi) and [SACN](https://github.com/JD-AI-Research-Silicon-Valley/SACN).


