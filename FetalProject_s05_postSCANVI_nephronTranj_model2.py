#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors


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


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_neighbors.h5ad")


# In[4]:


sc.pl.umap(adata, color = ["cellType_SCANVI"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False, palette = colors)


# In[5]:


adata_nephron = adata[adata.obs.cellType_SCANVI.isin(["NPC","PT","LOH","DCT","PEC","Podocyte","Int"])]


# In[6]:


absorbtion_prob = pd.read_csv("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/specification_absorbtion_probabilities.csv")
selection_prob = pd.read_csv("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/selection_absorbtion_probabilities.csv")
selfRenew_prob = pd.read_csv("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/reNew_absorbtion_probabilities.csv")
pseudoTime_prob = pd.read_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/pseudoTime.csv')

absProbs = pd.concat([absorbtion_prob, selection_prob, selfRenew_prob, pseudoTime_prob], axis=1)
print(absProbs)


# In[7]:


cellRankadata = sc.read_h5ad("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineage_40_30_clean.h5ad")
absProbs.index = cellRankadata.obs.index

#adata_nephron.obs["PEC_absorbtion"] = absProbs["PEC"]
adata_nephron.obs["DCT_absorbtion"] = absProbs["DCT"]
adata_nephron.obs["LOH_absorbtion"] = absProbs["LOH"] 
adata_nephron.obs["PT_absorbtion"] = absProbs["PT"] 
adata_nephron.obs["Podo_absorbtion"] = absProbs["Podocyte"] 
adata_nephron.obs["NPC"] = absProbs["NPC"] 
adata_nephron.obs["Differentiated Cell"] = absProbs["Differentiated Cell"] 
adata_nephron.obs["Glomerular"] = absProbs["Glomerular"] 
adata_nephron.obs["Tubule"] = absProbs["Tubule"]
adata_nephron.obs["LatentTime"] = absProbs["LatentTime"]
adata_nephron.obs["PseudoTime"] = absProbs["PseudoTime"]

Cosmx_cells_mask = (adata_nephron.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata_nephron.obs['tech'] != 'CosMx')

sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["Podo_absorbtion", "DCT_absorbtion", "LOH_absorbtion","PT_absorbtion"], cmap = "viridis_r", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["NPC", "Differentiated Cell"], cmap = "viridis_r", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["Glomerular", "Tubule"], cmap = "viridis_r", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["LatentTime", "PseudoTime"], cmap = "viridis_r", frameon = False)


# In[8]:


sc.pp.neighbors(adata_nephron, use_rep="X_scANVI", n_neighbors = neighbors)
sc.tl.umap(adata_nephron, min_dist=0.3)
sc.pl.umap(adata_nephron, color = ["cellType_SCANVI"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# # let's impute the absorbtion probabilities

# In[9]:


Cosmx_cells_mask = (adata_nephron.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata_nephron.obs['tech'] != 'CosMx')

CosMx_index = np.where(Cosmx_cells_mask)[0]
scRNA_index = np.where(scRNA_cells_mask)[0]

nn = NearestNeighbors(n_neighbors= neighbors, metric='euclidean')
nn.fit(adata_nephron.obsm["X_scANVI"][scRNA_cells_mask])

distances_all_to_non_Cosmx, indices_all_to_non_Cosmx = nn.kneighbors(adata_nephron.obsm["X_scANVI"][Cosmx_cells_mask])

mean_distances = np.mean(distances_all_to_non_Cosmx, axis=1)
plt.hist(mean_distances, bins=500, edgecolor='black')
plt.title('Mean Distances to 15 Nearest Neighbors for Cosmx Cells in scVI space')
plt.xlabel('Mean Distance')
plt.ylabel('Frequency')
plt.show()

# Create a dictionary with cell indices as keys and mean distances as values
mean_distance_dict = {index: distance for index, distance in zip(adata_nephron.obs_names[Cosmx_cells_mask], mean_distances)}

adata_nephron.obs['mean_distance'] = adata_nephron.obs_names.map(mean_distance_dict)
adata_nephron.obs['mean_distance'] = adata_nephron.obs['mean_distance'].fillna(float('0'))


# In[10]:


sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["Podo_absorbtion", "DCT_absorbtion", "LOH_absorbtion","PT_absorbtion"], cmap = "plasma", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["NPC", "Differentiated Cell"], cmap = "plasma", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["Glomerular", "Tubule"], cmap = "plasma", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["LatentTime", "PseudoTime"], cmap = "plasma", frameon = False)


# ### remove pseudoTime outliers

# In[11]:


print(adata_nephron)

adata_nephron.obs['PseudoTime'] = adata_nephron.obs['PseudoTime'].fillna(float('0'))
print(adata_nephron[adata_nephron.obs["PseudoTime"] < 0.8])
adata_nephron = adata_nephron[adata_nephron.obs["PseudoTime"] < 0.8]


# In[12]:


sc.pp.neighbors(adata_nephron, use_rep="X_scANVI", n_neighbors = neighbors)
sc.tl.umap(adata_nephron, min_dist=0.3)
sc.tl.leiden(adata_nephron)
sc.pl.umap(adata_nephron, color = ["cellType_SCANVI"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)
sc.pl.umap(adata_nephron, color = ["leiden"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)

Cosmx_cells_mask = (adata_nephron.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata_nephron.obs['tech'] != 'CosMx')

CosMx_index = np.where(Cosmx_cells_mask)[0]
scRNA_index = np.where(scRNA_cells_mask)[0]

nn = NearestNeighbors(n_neighbors= neighbors, metric='euclidean')
nn.fit(adata_nephron.obsm["X_scANVI"][scRNA_cells_mask])

distances_all_to_non_Cosmx, indices_all_to_non_Cosmx = nn.kneighbors(adata_nephron.obsm["X_scANVI"][Cosmx_cells_mask])


# In[13]:


sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["Podo_absorbtion", "DCT_absorbtion", "LOH_absorbtion","PT_absorbtion"], cmap = "plasma", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["NPC", "Differentiated Cell"], cmap = "plasma", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["Glomerular", "Tubule"], cmap = "coolwarm", frameon = False)
sc.pl.umap(adata_nephron[scRNA_cells_mask,:], color = ["LatentTime", "PseudoTime"], cmap = "plasma", frameon = False)


# ### Impute absorbtion

# In[14]:


absDataframe = adata_nephron.obs[["DCT_absorbtion", "LOH_absorbtion", "PT_absorbtion", "Podo_absorbtion", "NPC", "Tubule","Glomerular","Differentiated Cell", "LatentTime", "PseudoTime"]]

scRNA_df = absDataframe.iloc[scRNA_index].T

CosMx_df = absDataframe.iloc[CosMx_index]
CosMx_df.loc[:] = np.nan

adj = (distances_all_to_non_Cosmx ** -2.0).sum(axis=1)
affinity_array = (distances_all_to_non_Cosmx ** -2)

pd.options.mode.chained_assignment = None

for i in range(len(indices_all_to_non_Cosmx)):
      CosMx_df.iloc[i,:] = (scRNA_df.iloc[:,indices_all_to_non_Cosmx[i]] * affinity_array[i]).sum(axis=1)/adj[i]

adata_nephron.obs["DCT_absorbtion_SCVI"] = CosMx_df["DCT_absorbtion"]
adata_nephron.obs["LOH_absorbtion_SCVI"] = CosMx_df["LOH_absorbtion"]
adata_nephron.obs["PT_absorbtion_SCVI"] = CosMx_df["PT_absorbtion"]
adata_nephron.obs["Podo_absorbtion_SCVI"] = CosMx_df["Podo_absorbtion"]
adata_nephron.obs["NPC_SCVI"] = CosMx_df["NPC"]
adata_nephron.obs["Differentiated Cell_SCVI"] = CosMx_df["Differentiated Cell"]
adata_nephron.obs["Tubule_SCVI"] = CosMx_df["Tubule"]
adata_nephron.obs["Glomerular_SCVI"] = CosMx_df["Glomerular"]
adata_nephron.obs["LatentTime_SCVI"] = CosMx_df["LatentTime"]
adata_nephron.obs["PseudoTime_SCVI"] = CosMx_df["PseudoTime"]


# In[15]:


sc.pl.umap(adata_nephron[Cosmx_cells_mask,:], color = ["Glomerular_SCVI", "Tubule_SCVI"], cmap = "coolwarm", frameon = False)


# In[16]:


adata_sample = adata_nephron[adata_nephron.obs["sample"] == "1"]
sc.pl.scatter(
    adata_sample, 
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="Glomerular_SCVI",
    size=4,
    color_map="coolwarm"
)

sc.pl.scatter(
    adata_sample, 
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="LatentTime_SCVI",
    size=4,
    color_map="plasma"
)

sc.pl.scatter(
    adata_sample, 
    x="CenterX_global_px",
    y="CenterY_global_px",
    color="PseudoTime_SCVI",
    size=4,
    color_map="plasma"
)


# In[17]:


adata.obs["DCT_absorbtion_SCVI"] = adata_nephron.obs["DCT_absorbtion_SCVI"]
adata.obs["LOH_absorbtion_SCVI"] = adata_nephron.obs["LOH_absorbtion_SCVI"]
adata.obs["PT_absorbtion_SCVI"] = adata_nephron.obs["PT_absorbtion_SCVI"]
adata.obs["Podo_absorbtion_SCVI"] = adata_nephron.obs["Podo_absorbtion_SCVI"] 
adata.obs["NPC_SCVI"] = adata_nephron.obs["NPC_SCVI"] 
adata.obs["Differentiated Cell_SCVI"] = adata_nephron.obs["Differentiated Cell_SCVI"] 
adata.obs["Tubule_SCVI"] = adata_nephron.obs["Tubule_SCVI"]
adata.obs["Glomerular_SCVI"] = adata_nephron.obs["Glomerular_SCVI"] 
adata.obs["LatentTime_SCVI"] = adata_nephron.obs["LatentTime_SCVI"] 
adata.obs["PseudoTime_SCVI"] = adata_nephron.obs["PseudoTime_SCVI"] 

print(adata)

adata.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_Abs.h5ad")


# In[ ]:




