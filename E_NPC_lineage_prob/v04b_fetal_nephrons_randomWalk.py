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


adata = scv.read("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineage_40_30_clean.h5ad")


scv.pl.velocity_embedding_stream(adata, basis='umap', color = "cellType")


# In[4]:


from cellrank.tl.kernels import VelocityKernel

vk = VelocityKernel(adata)
vk.compute_transition_matrix()
print(vk)


# In[5]:


vk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT","PEC"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[6]:


from cellrank.tl.kernels import ConnectivityKernel

#using knn we will generate a connectivity kernel, this is from the knn graph based on PCs from the harmony calculation
print(adata)

ck = ConnectivityKernel(adata).compute_transition_matrix()
print(ck)


# In[7]:


ck.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT","PEC"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[8]:


ck.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[9]:


cbk = 0.8 * vk + 0.2 * ck
print(cbk)
cbk.compute_transition_matrix(adata)

cbk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[10]:


cbk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT","PEC"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[11]:


cbk = 0.5 * vk + 0.5 * ck
print(cbk)
cbk.compute_transition_matrix(adata)

cbk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT"]},
    successive_hits=10,
    cmap="viridis",
    max_iter=1000,
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[12]:


cbk = 0.5 * vk + 0.5 * ck
print(cbk)
cbk.compute_transition_matrix(adata)

cbk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT","PEC"]},
    successive_hits=5,
    cmap="viridis",
    max_iter=1000,
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[13]:


cbk = 0.2 * vk + 0.8 * ck
print(cbk)
cbk.compute_transition_matrix(adata)

cbk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[14]:


cbk = 0.2 * vk + 0.8 * ck
print(cbk)
cbk.compute_transition_matrix(adata)

cbk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT","PEC"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[15]:


vk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)


# In[16]:


vk.plot_random_walks(
    500,
    start_ixs={"cellType": "NPC"},
    stop_ixs={"cellType": ["Podocyte","PT","LOH","DCT","PEC"]},
    successive_hits=5,
    max_iter=1000,
    cmap="viridis",
    show_progress_bar=False,
    ixs_legend_loc="best",
    seed=42,
)

