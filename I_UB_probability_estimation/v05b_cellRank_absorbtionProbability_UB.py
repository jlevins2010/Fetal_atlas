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


adata = scv.read("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/UBLineage_40_30_clean.h5ad")

adata.obs["cellType"] =  adata.obs["cellType"].replace("Urethelium", "Urothelium")


scv.pl.velocity_embedding_stream(adata, basis='umap', color = "cellType", palette = "Dark2")


# # Generate kernel

# In[4]:


from cellrank.tl.kernels import VelocityKernel

vk = VelocityKernel(adata)
vk.compute_transition_matrix()
print(vk)

from cellrank.tl.kernels import ConnectivityKernel

#using knn we will generate a connectivity kernel, this is from the knn graph based on PCs from the harmony calculation
print(adata)

ck = ConnectivityKernel(adata).compute_transition_matrix()
print(ck)

# weighting of velocity vs connectivity
cbk = 0.5 * vk + 0.5 * ck
print(cbk)


# # Create Estimator

# In[5]:


from cellrank.tl.estimators import GPCCA

g = GPCCA(cbk)
print(g)

g.compute_schur(n_components=5)
g.plot_spectrum()

g.compute_macrostates(n_states=5, cluster_key="cellType")


#consider removing clusters, changing to cellType
g.plot_macrostates()
g.plot_macrostates(same_plot=False)
g.plot_macrostates(discrete=True)

max_values = pd.DataFrame(np.argmax(g.macrostates_memberships, axis=1))
max_values.index = adata.obs.index
adata.obs["Macrostate"] =  max_values


# # Set initial states as UB

# In[6]:


initial_states = (g.macrostates == 'UB_2')

adata.obs["initial_states_probs"] = g.macrostates_memberships["UB_2"]
adata.obs["initial_states"] = initial_states


# Set Terminal cell fates

# In[7]:


g.set_terminal_states({"CNT": adata[adata.obs["cellType"] == "CNT"].obs_names, \
                       "IC": adata[adata.obs["cellType"] == "IC"].obs_names, \
                       "Urothelium": adata[adata.obs["cellType"] == "Urothelium"].obs_names})


# In[8]:


g.compute_absorption_probabilities()

g.plot_coarse_T(text_kwargs={"fontsize": 10})

g.plot_absorption_probabilities(same_plot = False)

adata.uns['iroot'] = np.flatnonzero(adata.obs['initial_states'] == True)[0]
sc.tl.dpt(adata)

scv.tl.recover_latent_time(
    adata, root_key="initial_states_probs", end_key="terminal_states_probs"
)


# In[9]:


scv.tl.paga(
    adata,
    groups="cellType",
    root_key="initial_states_probs",
    end_key="terminal_states_probs",
    use_time_prior="dpt_pseudotime",
)

sc.set_figure_params(figsize=(5, 5))

cr.pl.cluster_fates(
    adata,
    mode="paga_pie",
    cluster_key="cellType",
    basis="umap",
    legend_kwargs={"loc": "bottom left"},
    legend_loc="on data",
    node_size_scale=5,
    edge_width_scale=1,
    max_edge_width=4,
    title="directed PAGA",
)


# In[10]:


df = pd.DataFrame(g.absorption_probabilities, columns = g.absorption_probabilities.names)

df.to_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/Absorption_probabilities_UBsubclusters.csv', index=False)

