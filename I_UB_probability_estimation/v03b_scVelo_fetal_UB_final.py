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


adata_nep = scv.read('/home/levinsj/Fetal_dir/Velocyto/02_mergedSCV/allFetal.h5ad', cache=True)

ldata = scv.read("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalUBOnly_annotated_final.h5ad")
adata_nep = scv.utils.merge(adata_nep, ldata)

scv.pp.filter_and_normalize(adata_nep, min_shared_counts=20, n_top_genes=2000, subset_highly_variable = False)
scv.pp.moments(adata_nep, n_pcs=40, n_neighbors=30)
scv.tl.recover_dynamics(adata_nep)
scv.tl.velocity(adata_nep, mode="dynamical", n_jobs=8)
scv.tl.velocity_graph(adata_nep)

scv.pl.velocity_embedding_stream(adata_nep, basis='umap', color = "leiden")
scv.pl.velocity_embedding_stream(adata_nep, basis='umap', color = "cellType")
scv.tl.velocity_pseudotime(adata_nep)

scv.pl.scatter(adata_nep, color='velocity_pseudotime', color_map='gnuplot')

scv.tl.velocity_confidence(adata_nep)
keys = 'velocity_length', 'velocity_confidence'
scv.pl.scatter(adata_nep, c=keys, cmap='coolwarm', perc=[5, 95])

adata_nep.write_h5ad(filename = "/home/levinsj/Fetal_dir/Velocyto/03_CellRank/UBLineage_40_30_clean.h5ad")

