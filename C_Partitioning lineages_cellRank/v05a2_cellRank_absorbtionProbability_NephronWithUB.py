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


sc.settings.set_figure_params(frameon=False, dpi_save=1000)
scv.set_figure_params('scvelo')


# In[3]:


adata = scv.read("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroAndUBLineage_40_30.h5ad")


scv.pl.velocity_embedding_stream(adata, basis='umap', color = "cellType")


# In[4]:


from cellrank.tl.kernels import VelocityKernel

vk = VelocityKernel(adata, backward=False)
vk.compute_transition_matrix()
print(vk)

from cellrank.tl.kernels import ConnectivityKernel

#using knn we will generate a connectivity kernel, this is from the knn graph based on PCs from the harmony calculation
print(adata)

ck = ConnectivityKernel(adata, backward=False).compute_transition_matrix()
print(ck)


# Let's see which cell populations are derived from which progenitors

# In[5]:


from cellrank.tl.estimators import GPCCA

vk = VelocityKernel(adata, backward=True).compute_transition_matrix()
ck = ConnectivityKernel(adata, backward=True).compute_transition_matrix()

cbk = 0.5 * vk + 0.5 * ck
g = GPCCA(cbk)
g.compute_schur(n_components=10)
g.compute_macrostates(n_states=10, cluster_key="cellType")

g.plot_macrostates(same_plot=False)
g.plot_macrostates(discrete=True)
g.set_terminal_states_from_macrostates(names=["NPC","UB_CT"], n_cells=200) # previous
g.compute_absorption_probabilities()
g.plot_absorption_probabilities(legend_loc = 'right margin', palette = "tab10")
g.plot_absorption_probabilities(legend_loc = 'right margin', palette = "Dark2")
g.plot_absorption_probabilities(legend_loc = 'right margin', palette = "Set1")
      
    
adata.obs["NPC"] = np.array(g.absorption_probabilities["NPC"])
adata.obs["UB_CT"] = np.array(g.absorption_probabilities["UB_CT"])
adata.obs["lineage"] = 'NA'
adata.obs["lineage"][adata.obs["NPC"] > 0.5] = "NPC"
adata.obs["lineage"][adata.obs["NPC"] < 0.5] = "UB"
sc.pl.umap(adata, color = "lineage")


# In[6]:


sc.pl.umap(adata, color = ["CLDN16","PTH1R","GATA3"], cmap='viridis_r')

adata_NPC = adata[adata.obs[g.absorption_probabilities["NPC"] >= 0.5].index]
adata_UB = adata[adata.obs[g.absorption_probabilities["UB_CT"] > 0.5].index]
sc.pl.umap(adata_NPC, color = "leiden")
sc.pl.umap(adata_UB, color = "leiden")

sc.pl.umap(adata_NPC, color = ["CLDN16","PTH1R","GATA3"], cmap='viridis_r')
sc.pl.umap(adata_UB, color = ["CLDN16","PTH1R","GATA3"], cmap='viridis_r')

sc.pl.umap(adata_NPC, color = ["ATP6V1G3","SLC4A1","SLC26A4"], cmap='viridis_r')
sc.pl.umap(adata_UB, color = ["ATP6V1G3","SLC4A1","SLC26A4"], cmap='viridis_r')


df = pd.DataFrame(g.absorption_probabilities, columns = g.absorption_probabilities.names)

df.to_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/Absorption_probabilities_nephron_UB_a2.csv', index=False)

