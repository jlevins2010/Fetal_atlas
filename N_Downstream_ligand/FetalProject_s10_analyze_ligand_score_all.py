#!/usr/bin/env python
# coding: utf-8

# In[1]:


import anndata as ad
import scanpy as sc
import squidpy as sq
import pandas as pd
from scipy.sparse import csr_matrix # imports the csr_matrix function from the scipy.sparse module


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# In[2]:


mpl.rcParams['figure.dpi'] = 450


# Set parameters

# In[3]:


neighbors = 15
fineNeighborhood = 4
coarseNeighborhood = 3

colors = {"DCT": "#800515",
               "Endothelium": "#7ae031",
               "UB_CT": "black",
               "Podocyte": "#ad9c00", 
               "Stroma": "#794b82",
               "NPC": "#ff8000", 
               "PT": "#ff00d4", 
               "Int": "#698cff",
               "Ureth": "#d47222", 
               "PEC": "#ff0011", 
               "LOH": "#235e00",
               "Immune Cells": '#757575',
               "Nephron":"#698cff",
         }


# In[4]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
print(adata)


# In[5]:


ligand_score = pd.read_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK1_all_ligand_scores.csv', index_col="Unnamed: 0").T
print(ligand_score.columns)
print(ligand_score.columns.duplicated())
ligand_score = ligand_score.loc[:,~ligand_score.columns.duplicated()].copy()
print(ligand_score.columns)
adata1 = ad.AnnData(X = csr_matrix(ligand_score.values))
adata1.var_names = ligand_score.columns.tolist()
adata1.obs_names = ligand_score.index.tolist()
adata1.obs["sample"] = "FK1"
adata1.obs = adata1.obs.astype(str)
adata1.var = adata1.var.astype(str)

ligand_score = pd.read_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK4_all_ligand_scores.csv', index_col="Unnamed: 0").T
ligand_score = ligand_score.loc[:,~ligand_score.columns.duplicated()].copy()
adata2 = ad.AnnData(X = csr_matrix(ligand_score.values))
adata2.var_names = ligand_score.columns.tolist()
adata2.obs_names = ligand_score.index.tolist()
adata2.obs["sample"] = "FK4"
adata2.obs = adata2.obs.astype(str)
adata2.var = adata2.var.astype(str)

ligand_score = pd.read_csv('/home/levinsj/spatial/adata/Export_to_R_files/HK3524_all_ligand_scores.csv', index_col="Unnamed: 0").T
ligand_score = ligand_score.loc[:,~ligand_score.columns.duplicated()].copy()
adata3 = ad.AnnData(X = csr_matrix(ligand_score.values))
adata3.var_names = ligand_score.columns.tolist()
adata3.obs_names = ligand_score.index.tolist()
adata3.obs["sample"] = "HK3524"
adata3.obs = adata3.obs.astype(str)
adata3.var = adata3.var.astype(str)


# In[6]:


samples = [adata1, adata2, adata3]


# In[7]:


print(samples)


# In[8]:


for i in samples:
    i = i.obs_names_make_unique()

print(samples)


# In[9]:


adata_merge = ad.concat(samples, axis=0, join='inner')
print(adata_merge)


# In[10]:


print(adata_merge.var_names)


# In[11]:


print(adata_merge.obs_names[0:10])


# In[12]:


sc.pp.pca(adata_merge)
sc.pp.neighbors(adata_merge, n_neighbors = 50, n_pcs=30)
sc.tl.umap(adata_merge, min_dist = 0.3)
sc.pl.umap(adata_merge, color = "sample")


# In[13]:


sc.pl.umap(adata_merge, color = ["VEGFA","ACE2","WNT4","GDNF"], cmap = "viridis_r", legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[14]:


adata_merge.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/ligandScores_all_ligands.h5ad")


# In[15]:


adata0 = adata[adata.obs["sample"] == "0"]
adata1 = adata[adata.obs["sample"] == "1"]
adata2 = adata[adata.obs["sample"] == "2"]

print(adata0)
print(adata1)
print(adata2)

