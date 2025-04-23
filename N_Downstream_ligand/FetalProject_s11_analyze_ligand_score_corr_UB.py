#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix # imports the csr_matrix function from the scipy.sparse module
import matplotlib.colors as mcolors

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


# import all CosMx UB cells with imputed absorption probabilities


# In[5]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_Abs_UB_Only_final.h5ad")

print(adata.obs["CNT_absorption"])
adata = adata[adata.obs["UB_absorption_SCVI"] > 0.5]

print(adata.obs["CNT_absorption"])


# In[6]:


Cosmx_cells_mask = (adata.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata.obs['tech'] != 'CosMx')

sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["CNT_absorption_SCVI", "Urethelium_absorption_SCVI","IC_absorption_SCVI"], cmap = "coolwarm", frameon = False)


# In[7]:


adata1 = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/ligandScores_all_ligands.h5ad")


# In[8]:


print(adata1.obs_names)


# In[9]:


adata1.obs["CNT_absorption"] = adata.obs["CNT_absorption_SCVI"]
adata1.obs["IC_absorption"] = adata.obs['IC_absorption_SCVI']
adata1.obs["Urethelium_absorption"] = adata.obs["Urethelium_absorption_SCVI"]



adata1 = adata1[adata1.obs["CNT_absorption"].notna()]
print(adata1)
print(adata1.obs["CNT_absorption"])


# In[10]:


Cnt_correlations = []
Ic_correlations = []
Ureth_correlations = []


for i in range(adata1.shape[1]):
    gene_expression = adata1.X[:, i].toarray().flatten()
    gene_expression = gene_expression[~np.isnan(gene_expression)]
    cnt = adata1.obs["CNT_absorption"][~np.isnan(gene_expression)]
    ic = adata1.obs["IC_absorption"][~np.isnan(gene_expression)]
    ure = adata1.obs["Urethelium_absorption"][~np.isnan(gene_expression)]

    # Calculate correlations
    correlation_cnt = np.corrcoef(gene_expression, cnt)[0, 1]
    correlation_ic = np.corrcoef(gene_expression, ic)[0, 1]
    correlation_ure = np.corrcoef(gene_expression, ure)[0, 1]
    
    Cnt_correlations.append(correlation_cnt)
    Ic_correlations.append(correlation_ic)
    Ureth_correlations.append(correlation_ure)
    

df = pd.DataFrame({'Ligand': adata1.var_names.tolist(), 'CNT': Cnt_correlations, "IC": Ic_correlations,
                  "Urethelium": Ureth_correlations})

print(df)
df.to_csv('/home/levinsj/spatial/adata/project_Files/Fetal/ligandCorrUB.csv')

