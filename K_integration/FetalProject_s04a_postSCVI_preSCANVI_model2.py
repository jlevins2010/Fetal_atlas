#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors


# Set parameters

# In[2]:


neighbors = 15


# load model 6

# In[3]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_annotated_AllCells_model2_postSCVI.h5ad")


# In[4]:


sc.pl.umap(adata, color = "tech", frameon = False)


# In[5]:


sc.pl.umap(adata, color = ["cellType","cellType_CosMx_1", "cellType3"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# get cell IDs for each technology

# In[6]:


from sklearn.neighbors import NearestNeighbors

# Specify the number of neighbors (e.g., 15)
n_neighbors = 15

# Filter cells where ["tech"] is 'Cosmx'
Cosmx_cells_mask = (adata.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata.obs['tech'] != 'CosMx')

CosMx_index = np.where(Cosmx_cells_mask)[0]
scRNA_index = np.where(scRNA_cells_mask)[0]

nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
nn.fit(adata.obsm["X_scVI"][scRNA_cells_mask])

distances_all_to_non_Cosmx, indices_all_to_non_Cosmx = nn.kneighbors(adata.obsm["X_scVI"][Cosmx_cells_mask])

print(len(indices_all_to_non_Cosmx))


# In[7]:


mean_distances = np.mean(distances_all_to_non_Cosmx, axis=1)
plt.hist(mean_distances, bins=500, edgecolor='black')
plt.title('Mean Distances to 15 Nearest Neighbors for Cosmx Cells in scVI space')
plt.xlabel('Mean Distance')
plt.ylabel('Frequency')
plt.show()


# In[8]:


# Create a dictionary with cell indices as keys and mean distances as values
mean_distance_dict = {index: distance for index, distance in zip(adata.obs_names[Cosmx_cells_mask], mean_distances)}

# Update the 'mean_distance' column in the 'obs' attribute of adata
adata.obs['mean_distance'] = adata.obs_names.map(mean_distance_dict)
adata.obs['mean_distance'] = adata.obs['mean_distance'].fillna(float('inf'))

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 40)

print(f'Threshold for the lowest 40% of distances: {percentile_threshold}')


# In[9]:


sc.pl.umap(adata, color = ["mean_distance"], cmap = "viridis_r", frameon = False)

adata.obs['mean_distance'] = adata.obs_names.map(mean_distance_dict)
adata.obs['mean_distance'] = adata.obs['mean_distance'].fillna(float('0'))

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 50)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 50")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 60)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 40")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 70)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 30")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 75)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 25")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 80)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 20")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 85)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 15")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 90)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 10")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 95)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 5")

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 99)
adata.obs["worst_pct"] = (adata.obs['mean_distance'] > percentile_threshold).astype(str)
sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["worst_pct"], frameon = False, title = "Worst 1")


# In[10]:


sc.pp.neighbors(adata, use_rep="X_scVI")

sc.tl.umap(adata)

sc.pl.umap(adata, color = ["cellType","cellType_CosMx_1", "cellType3"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[11]:


sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["cellType_CosMx_1"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)
sc.pl.umap(adata[scRNA_index,:], color = ["cellType"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[12]:


print(adata.obs.iloc[:,4].value_counts())
print(adata.obs.iloc[:,20].describe())


# In[13]:


cellTypeCol = adata.obs.columns.get_loc("cellType3")
distCol = adata.obs.columns.get_loc("mean_distance")


# In[14]:


scRNA_cell_types = adata[scRNA_index, :].obs.cellType3.values

percentile_threshold = np.percentile(adata[CosMx_index, :].obs['mean_distance'], 60)
print(percentile_threshold)

for i in CosMx_index:
    distance = adata.obs.iloc[i, distCol] # column 20 of adata.obs should be distance
    if distance < percentile_threshold:
        neighbor_index = indices_all_to_non_Cosmx[i]
        unique, counts = np.unique((scRNA_cell_types[neighbor_index]), return_counts=True)
        adata.obs.iloc[i, cellTypeCol] = unique[np.argmax(counts)] # column 4 should be your cellType annotation from snRNAseq


# In[15]:


adata = adata[adata.obs["mean_distance"] < np.percentile(adata[CosMx_index, :].obs['mean_distance'], 80)]


# In[16]:


adata.obs["cellType3"].value_counts()


# In[17]:


Cosmx_cells_mask = (adata.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata.obs['tech'] != 'CosMx')

CosMx_index = np.where(Cosmx_cells_mask)[0]
scRNA_index = np.where(scRNA_cells_mask)[0]

adata[Cosmx_cells_mask,:].obs["cellType3"].value_counts()


# In[18]:


sc.pl.umap(adata[Cosmx_cells_mask,:], color = ["cellType","cellType_CosMx_1", "cellType3"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)
sc.pl.umap(adata[scRNA_index,:], color = ["cellType","cellType_CosMx_1", "cellType3"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[19]:


adata.obs['pct_counts_mt'] = adata.obs['pct_counts_mt'].fillna(0)


# In[20]:


adata.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_annotated_AllCells_model2_preSCANVI.h5ad")

