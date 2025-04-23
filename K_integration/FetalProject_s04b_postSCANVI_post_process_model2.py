#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors

mpl.rcParams['figure.dpi'] = 450


# Set parameters

# In[2]:


neighbors = 15

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
         }


# In[3]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_annotated_AllCells_model2_postSCANVI.h5ad")


# adata.X = adata.layers["counts"] # set X-layer prior to subset
# adata = adata[~adata.obs.leiden.isin(["28"])]

# In[4]:


sc.pl.umap(adata, color = ["cellType","cellType_CosMx_1"],  frameon = False)
sc.pl.umap(adata, color = ["cellType3"], frameon = False)
sc.pl.umap(adata, color = ["tech"], legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[5]:


cell_identities = {'0': 'Int', '1':"Stroma", '2': 'Int', '3': 'Stroma', '4': 'PT', '5': 'Podocyte', '6':'Int','7':'Endothelium','8':'Int',
                   '9': 'Podocyte', '10': 'Stroma', '11':'UB_CT','12':'DCT','13':'Int','14':'Stroma','15':'LOH','16':'Endothelium','17':'Int',
                   '18':'Stroma','19':'PEC','20':'PT','21':'Int','22':'UB_CT','23':'Immune Cells','24':'Ureth',
                   '25':'Int','26':'Int','27':'PT','28':"Endothelium", '29':"DCT"}

adata.obs["cellType_SCANVI"] = adata.obs['leiden'].map(cell_identities).astype('category')

sc.pl.umap(adata, color = ["cellType_SCANVI"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False, palette = colors)


# In[6]:


sc.tl.leiden(adata, restrict_to=('leiden', ['18']), resolution=0.1, key_added='leiden_sub18')

sc.pl.umap(adata, color = "leiden_sub18", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[7]:


# Filter cells where ["tech"] is 'Cosmx'
Cosmx_cells_mask = (adata.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata.obs['tech'] != 'CosMx')

sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["cellType_SCANVI"], frameon = False, palette = colors)

CosMx_index = np.where(Cosmx_cells_mask)[0]
scRNA_index = np.where(scRNA_cells_mask)[0]

#sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["cellType_SCANVI", "cellType_CosMx_1","cellType3"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)
#sc.pl.umap(adata[scRNA_index,:], color = ["cellType_SCANVI", "cellType_CosMx_1","cellType3"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)
#sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["mean_distance"], frameon = False)
print(len(CosMx_index))
print(adata.obs["sample"].value_counts())


# In[8]:


sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["cellType_CosMx_1"],  frameon = False)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["mean_distance"],  frameon = False)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["mean_distance"],  frameon = False)


# In[9]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 3))
sc.pl.umap(adata[Cosmx_cells_mask,:], color = "sample", groups = "0",frameon = False, legend_loc = False, title = "FK0", show = False, ax = axes[0])
sc.pl.umap(adata[Cosmx_cells_mask,:], color = "sample", groups = "1",frameon = False, legend_loc = False, title = "FK1", show = False, ax = axes[1])
sc.pl.umap(adata[Cosmx_cells_mask,:], color = "sample", groups = "2",frameon = False, legend_loc = False, title = "FK4", show = False, ax = axes[2])
plt.axis('off')

plt.show()


# In[10]:


plt.hist(adata[Cosmx_cells_mask,:].obs["mean_distance"], bins=500, edgecolor='black')
plt.title('Mean Distances to 15 Nearest Neighbors for Cosmx Cells in scVI space')
plt.xlabel('Mean Distance')
plt.ylabel('Frequency')
plt.show()


# In[11]:


print(adata[Cosmx_cells_mask,:].n_obs)


# In[12]:


adata_cosMx = adata[Cosmx_cells_mask,:]

adata_cosMx.X = adata_cosMx.layers["counts"]
sc.pp.normalize_total(adata_cosMx, target_sum=1e4)
sc.pp.log1p(adata_cosMx)

sc.tl.rank_genes_groups(adata_cosMx, 'cellType_SCANVI', method='wilcoxon')
sc.pl.rank_genes_groups(adata_cosMx, n_genes=25, sharey=False)


# In[13]:


print(adata)


# In[14]:


adata.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_annotated_AllCells_model2_postSCANVI_anno.h5ad")


# In[15]:


adata_cosMx = adata[adata.obs["tech"] == "CosMx"]

for i in adata_cosMx.obs["sample"].unique():
    adata_cosMx_sample = adata_cosMx[adata_cosMx.obs["sample"] == i]
    sc.pl.scatter(
        adata_cosMx_sample,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="cellType_SCANVI",
        size=2,
    )
    sc.pl.scatter(
        adata_cosMx_sample,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="cellType_SCANVI",
        size=3,
    )
    sc.pl.scatter(
        adata_cosMx_sample,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="cellType_SCANVI",
        size=4,
    )


# In[16]:


adata_stroma = adata[adata.obs["cellType_SCANVI"] == "Stroma"]
adata_stroma.X = adata_stroma.layers["counts"]

sc.pp.normalize_total(adata_stroma, target_sum=1e4)
sc.pp.log1p(adata_stroma)

sc.tl.rank_genes_groups(adata_stroma, 'leiden_sub18', method='wilcoxon')
sc.pl.rank_genes_groups(adata_stroma, n_genes=25, sharey=False)


# In[17]:


adata_cosMx = adata_stroma[adata_stroma.obs["tech"] == "CosMx"]

for i in adata_cosMx.obs["sample"].unique():
    adata_cosMx_sample = adata_cosMx[adata_cosMx.obs["sample"] == i]
    sc.pl.scatter(
        adata_cosMx_sample,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_sub18",
        size=3,
    )

