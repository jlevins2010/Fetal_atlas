#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors


# Set parameters

# In[2]:


neighbors = 15
radius1 = 25
radius2 = 100

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


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_annotated_AllCells_model2_postSCANVI_anno.h5ad")


# In[4]:


print(adata)


# ### Let's subset to sample 1

# In[5]:


adata_sample = adata[adata.obs["sample"] == "1"]
    
sq.gr.spatial_neighbors(
    adata_sample,
    radius=radius1/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "fine_spatial"
)

print(adata_sample)

sq.gr.spatial_neighbors(
    adata_sample,
    radius=radius2/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "coarse_spatial"
)

print(adata_sample)



# In[6]:


neighbor_check = []
for k in range(adata_sample.obsp["fine_spatial_connectivities"].shape[0]):
    neighbor_check.append(np.sum(adata_sample.obsp["fine_spatial_connectivities"][k]))
    # total number of neighbors
plt.hist(neighbor_check, bins = np.arange(0, 50, 1))
plt.show()

adata_sample.obs["neighbors"] = 0
for i in range(adata_sample.obsp["fine_spatial_connectivities"].shape[0]):
    adata_sample.obs["neighbors"].iloc[i] = np.sum(adata_sample.obsp["fine_spatial_connectivities"][i])

sc.pl.scatter(
    adata_sample,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="neighbors",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

adata_sample = adata_sample[adata_sample.obs["neighbors"] > 2]

sc.pl.scatter(adata_sample,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="neighbors",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

sq.gr.nhood_enrichment(adata_sample, cluster_key="cellType_SCANVI", connectivity_key = "fine_spatial")
sq.pl.nhood_enrichment(
    adata_sample,
    cluster_key="cellType_SCANVI",
    figsize=(5, 5),
    title="Neighborhood enrichment adata",
    
)

adata_sample.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors_sample1.h5ad")


# ### sample 0

# In[7]:


adata_sample = adata[adata.obs["sample"] == "0"]
    
sq.gr.spatial_neighbors(
    adata_sample,
    radius=radius1/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "fine_spatial"
)

print(adata_sample)

sq.gr.spatial_neighbors(
    adata_sample,
    radius=radius2/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "coarse_spatial"
)

neighbor_check = []
for k in range(adata_sample.obsp["fine_spatial_connectivities"].shape[0]):
    neighbor_check.append(np.sum(adata_sample.obsp["fine_spatial_connectivities"][k]))
    # total number of neighbors
plt.hist(neighbor_check, bins = np.arange(0, 50, 1))
plt.show()

adata_sample.obs["neighbors"] = 0
for i in range(adata_sample.obsp["fine_spatial_connectivities"].shape[0]):
    adata_sample.obs["neighbors"].iloc[i] = np.sum(adata_sample.obsp["fine_spatial_connectivities"][i])

sc.pl.scatter(
    adata_sample,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="neighbors",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

adata_sample = adata_sample[adata_sample.obs["neighbors"] > 2]

sc.pl.scatter(adata_sample,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="neighbors",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

sq.gr.nhood_enrichment(adata_sample, cluster_key="cellType_SCANVI", connectivity_key = "fine_spatial")
sq.pl.nhood_enrichment(
    adata_sample,
    cluster_key="cellType_SCANVI",
    figsize=(5, 5),
    title="Neighborhood enrichment adata",
    
)

adata_sample.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors_sample0.h5ad")


# ### sample 2

# In[8]:


adata_sample = adata[adata.obs["sample"] == "2"]
    
sq.gr.spatial_neighbors(
    adata_sample,
    radius=radius1/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "fine_spatial"
)

sq.gr.spatial_neighbors(
    adata_sample,
    radius=radius2/0.12,
    coord_type="generic",
    spatial_key = "spatial_fov",
    key_added = "coarse_spatial"
)


neighbor_check = []
for k in range(adata_sample.obsp["fine_spatial_connectivities"].shape[0]):
    neighbor_check.append(np.sum(adata_sample.obsp["fine_spatial_connectivities"][k]))
    # total number of neighbors
plt.hist(neighbor_check, bins = np.arange(0, 50, 1))
plt.show()

adata_sample.obs["neighbors"] = 0
for i in range(adata_sample.obsp["fine_spatial_connectivities"].shape[0]):
    adata_sample.obs["neighbors"].iloc[i] = np.sum(adata_sample.obsp["fine_spatial_connectivities"][i])

sc.pl.scatter(
    adata_sample,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="neighbors",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)

adata_sample = adata_sample[adata_sample.obs["neighbors"] > 2]

sc.pl.scatter(adata_sample,
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="neighbors",
    size=1,
    legend_loc = 'on data', legend_fontsize=6,
    legend_fontoutline=2
)


sq.gr.nhood_enrichment(adata_sample, cluster_key="cellType_SCANVI", connectivity_key = "fine_spatial")
sq.pl.nhood_enrichment(
    adata_sample,
    cluster_key="cellType_SCANVI",
    figsize=(5, 5),
    title="Neighborhood enrichment adata",
    
)

adata_sample.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors_sample2.h5ad")


# In[9]:


adata0 = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors_sample0.h5ad")
adata1 = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors_sample1.h5ad")
adata2 = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors_sample2.h5ad")


# In[10]:


adata_list = [adata0, adata1, adata2]

for j in adata_list:
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="Endothelium")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["Endothelium_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["Endothelium_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
        
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="UB_CT")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["UB_CT_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["UB_CT_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
        
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="Stroma")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["Stroma_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["Stroma_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
        
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="Podocyte")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["Podocyte_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["Podocyte_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
        
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="PT")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["PT_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["PT_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
        
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="PEC")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["PEC_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["PEC_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
        
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="LOH")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["LOH_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["LOH_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
    
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="Int")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["Int_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["Int_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
    
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']=="DCT")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["DCT_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["DCT_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
        
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']== "Immune Cells")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["Immune Cells_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["Immune Cells_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
    
    cellNeighbors = (np.where(j.obs['cellType_SCANVI']== "Ureth")[0])
    cellNeighbors_location = np.where(j.obsp["fine_spatial_connectivities"][:, cellNeighbors].sum(axis=1) > 0)[0]
    j.obs["Ureth_neighbor"] = 0
    for i in cellNeighbors_location:
        j.obs["Ureth_neighbor"].iloc[i] = np.sum(j.obsp["fine_spatial_connectivities"][i, cellNeighbors])
      


# In[11]:


adata_scRNA = adata[adata.obs["tech"] == "scRNA"]


# In[12]:


adata = ad.concat([adata0, adata1, adata2, adata_scRNA], pairwise = True, join = "outer")
print(adata)

adata.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors.h5ad")

