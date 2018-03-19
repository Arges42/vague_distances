# Reasoning with Imprecise Temporal and Geographical Data

This repository contains the implementation of my master thesis. 

## What it is

The package **vague_distances** provides wrapper classes for the representation of uncertain spatial and temporal data.
After transforming the data into the vague representation all introduced distance metrics can be applied.
Additionally, the package also provides an interface for the computation of skylines using the vague representations.

## How to install

### Prerequisites:
1. scipy
2. pandas
3. pypref
4. pyemd

Installation with pip:
`pip install -r requirements.txt`

Additional (if the util package is used):
1. sshtunnel
2. pymongo

### Package:
`python setup.py install`
