#!/usr/bin/env python
# coding: utf-8

# # Importing data
# Let's start with importing our packages. We will use scanpy, scvelo and cell bender. We will run this with a GPU to boost efficiency.

# In[ ]:


get_ipython().system('pip install git+https://github.com/broadinstitute/CellBender.git')


# In[ ]:


# install packages

import os, sys
from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('pip install --quiet scvi-colab')
from scvi_colab import install
install()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scvi
import scanpy as sc


# # Running Cellbender
# Running cellBender on following samples. Requires GPU given --cuda argument.

# In[ ]:


#sampleRawH5Matricies = ["/content/drive/MyDrive/SusztakLabFiles/scRNAseq/RawData/HK2770_SC/raw_feature_bc_matrix.h5", 
                        "/content/drive/MyDrive/SusztakLabFiles/scRNAseq/RawData/HK2833_SN/raw_feature_bc_matrix.h5", 
                        "/content/drive/MyDrive/SusztakLabFiles/scRNAseq/RawData/HK2896_SN/raw_feature_bc_matrix.h5"]
sampleRawH5Matricies = ["/content/drive/MyDrive/SusztakLabFiles/multiOme/RawData/HK2725_SN/raw_feature_bc_matrix.h5"]
#sampleNames = ["HK2770_SC", "HK2833_SN", "HK2896_SN"]
sampleNames = ["HK2725"]

for index, item in enumerate(sampleRawH5Matricies):
    outFile="/content/drive/MyDrive/SusztakLabFiles/scRNAseq/cellBenderOutput/"+sampleNames[index]+"_output.h5"
    !cellbender  remove-background \
        --input {item} \
        --output {outFile} \
        --expected-cells 10000 \
        --total-droplets-included 50000 \
        --fpr 0.01 \
        --cuda \
        --epochs 150


# # Convert to Anndata object
# Now, let's convert this into a anndata object. We will then save it as a .h5ad for importing into scanpy or seurat.

# In[ ]:


for index, item in enumerate(sampleRawH5Matricies):
    outFile="/content/drive/MyDrive/SusztakLabFiles/scRNAseq/cellBenderOutput/"+sampleNames[index]+"_output_filtered.h5"
    adata = sc.read_10x_h5(outFile, genome="hg19")
    adata.write("/content/drive/MyDrive/SusztakLabFiles/scRNAseq/cellBenderOutput/"+sampleNames[index]+'_cellBender.h5ad')
    print(adata)

