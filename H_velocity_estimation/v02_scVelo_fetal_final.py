#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
import anndata
import scanpy as sc
import scvelo as scv
import cellrank as cr


# In[2]:


scv.set_figure_params('scvelo')


# In[3]:


adata_all = scv.read('/home/levinsj/Fetal_dir/Velocyto/02_mergedSCV/allFetal.h5ad', cache=True)
scv.pl.proportions(adata_all)
ldata = scv.read('/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad')


# In[4]:


adata = scv.utils.merge(adata_all, ldata)

scv.pl.proportions(adata)


# In[5]:


scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, subset_highly_variable = False)
scv.pp.moments(adata, n_pcs=40, n_neighbors=30)
scv.tl.velocity(adata, mode="dynamical", n_jobs=8)
scv.tl.velocity_graph(adata)

scv.pl.velocity_embedding_stream(adata, basis='umap', color = "cellType")


# In[6]:


scv.tl.velocity_confidence(adata)
keys = 'velocity_length', 'velocity_confidence'
scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95])

