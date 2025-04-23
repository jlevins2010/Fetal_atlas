#!/usr/bin/env python
# coding: utf-8

# In[1]:


import anndata as ad
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix # imports the csr_matrix function from the scipy.sparse module


import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# Set parameters

# In[2]:


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


# In[3]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
print(adata)


# In[4]:


ligand_score = pd.read_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK1_ligand_scores_conserved_list.csv', index_col="Unnamed: 0").T
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

ligand_score = pd.read_csv('/home/levinsj/spatial/adata/Export_to_R_files/FK4_ligand_scores_conserved_list.csv', index_col="Unnamed: 0").T
ligand_score = ligand_score.loc[:,~ligand_score.columns.duplicated()].copy()
adata2 = ad.AnnData(X = csr_matrix(ligand_score.values))
adata2.var_names = ligand_score.columns.tolist()
adata2.obs_names = ligand_score.index.tolist()
adata2.obs["sample"] = "FK4"
adata2.obs = adata2.obs.astype(str)
adata2.var = adata2.var.astype(str)

ligand_score = pd.read_csv('/home/levinsj/spatial/adata/Export_to_R_files/HK3524_ligand_scores_conserved_list.csv', index_col="Unnamed: 0").T
ligand_score = ligand_score.loc[:,~ligand_score.columns.duplicated()].copy()
adata3 = ad.AnnData(X = csr_matrix(ligand_score.values))
adata3.var_names = ligand_score.columns.tolist()
adata3.obs_names = ligand_score.index.tolist()
adata3.obs["sample"] = "HK3524"
adata3.obs = adata3.obs.astype(str)
adata3.var = adata3.var.astype(str)


# In[5]:


samples = [adata1, adata2, adata3]


# In[6]:


print(samples)


# In[7]:


for i in samples:
    i = i.obs_names_make_unique()

print(samples)


# In[8]:


adata_merge = ad.concat(samples, axis=0, join='inner')
print(adata_merge)


# In[9]:


print(adata_merge.var_names[0:159])


# In[10]:


print(adata_merge.obs_names[0:10])


# In[11]:


print(adata_merge.obs["sample"].value_counts())


# In[12]:


adata_merge.obs["renew"] = adata.obs["NPC_SCVI"]
adata_merge.obs["abs_PT"] = adata.obs["PT_absorbtion_SCVI"]
adata_merge.obs["abs_LOH"] = adata.obs["LOH_absorbtion_SCVI"]
adata_merge.obs["abs_DCT"] = adata.obs["DCT_absorbtion_SCVI"]
adata_merge.obs["abs_Podo"] = adata.obs["Podo_absorbtion_SCVI"]
adata_merge.obs["pseudoTime"] = adata.obs["PseudoTime_SCVI"]

adata_merge.obs["CenterY_global_px"] = adata.obs["CenterY_global_px"]
adata_merge.obs["CenterX_global_px"] = adata.obs["CenterX_global_px"]


# In[13]:


color_palette = [
    "#FF0000",  # Red - 1
    "#FFA500",  # Orange - 2
    "#FFFF00",  # Yellow - 3
    "#008000",  # Green - 4
    "#0000FF",  # Blue - 5
    "#800080",  # Purple - 6
    "#FFC0CB",  # Pink - 7
    "#804000",  # Brown - 8
    "#000000",  # Gray - 9
    "#000080",  # Navy - 9
    "#008080",  # Teal - 10
    "#808000",  # Olive - 11
    "#00FF00",  # Lime
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#800000",  # Maroon
    "#4B0082",  # Indigo
    "#EE82EE",  # Violet
    "#FFD700",  # Gold
    "#C0C0C0",  # Silver
    "#422436"   # White
]


# In[14]:


sc.pp.pca(adata_merge)
sc.pp.neighbors(adata_merge, n_neighbors = 15, n_pcs=30)
sc.tl.umap(adata_merge, min_dist = 0.4)
sc.tl.leiden(adata_merge, resolution = 0.3)
sc.pl.umap(adata_merge, color = "sample", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False)
sc.pl.umap(adata_merge, color = "leiden", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False, palette = color_palette)
sc.pl.umap(adata_merge, color = ["abs_PT","abs_Podo","abs_LOH","abs_DCT"], cmap = "viridis_r", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False)
sc.pl.umap(adata_merge, color = ["renew","pseudoTime"], cmap = "viridis_r", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False)


# adata_merge.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/ligandScores_conserved_ligands.h5ad")
# 

# sc.tl.rank_genes_groups(adata_merge, groupby='leiden', method='wilcoxon', pts = True)
# sc.pl.rank_genes_groups(adata_merge, n_genes=25, sharey=False)

# In[15]:


print(adata_merge.obs["sample"].value_counts())


# In[16]:


adata.obs["leiden_neighborhood"] = adata_merge.obs["leiden"]
adata.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression_neighborhoods.h5ad")
adata_merge.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_ligands.h5ad")



# In[17]:


sc.pl.umap(adata_merge, color = ["VEGFA","AVP","GDNF"], cmap = "viridis_r", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False)


# In[18]:


adata0 = adata_merge[adata_merge.obs["sample"] == "HK3524"]
adata1 = adata_merge[adata_merge.obs["sample"] == "FK1"]
adata2 = adata_merge[adata_merge.obs["sample"] == "FK4"]

print(adata_merge.obs["leiden"].value_counts())

print(adata0)
print(adata1.obs["leiden"].value_counts())
print(adata2)


# import scvelo as scv
# import matplotlib.pyplot as plt
# import numpy as np
# 
# plt.style.use('dark_background')  # Use dark background style

# In[19]:


size = 10


# scv.pl.scatter(
#         adata1,
#         x="CenterX_global_px",
#         y="CenterY_global_px",
#         color="AVP",
#         size= size,
#         legend_loc = 'none', legend_fontsize=6,
#         legend_fontoutline=2,
#         frameon = False, color_map = "plasma")

# scv.pl.scatter(
#         adata1,
#         x="CenterX_global_px",
#         y="CenterY_global_px",
#         color="WNT4",
#         size=size,
#         legend_loc = 'none', legend_fontsize=6,
#         legend_fontoutline=2,
#         frameon = False, color_map = "plasma", vmax = 0.2)

# scv.pl.scatter(
#         adata1,
#         x="CenterX_global_px",
#         y="CenterY_global_px",
#         color="GDNF",
#         size=size,
#         legend_loc = 'none', legend_fontsize=6,
#         legend_fontoutline=2,
#         frameon = False, color_map = "plasma")

# scv.pl.scatter(
#         adata1,
#         x="CenterX_global_px",
#         y="CenterY_global_px",
#         color="IGF1",
#         size=size,
#         legend_loc = 'none', legend_fontsize=6,
#         legend_fontoutline=2,
#         frameon = False, color_map = "plasma")

# scv.pl.scatter(
#         adata1,
#         x="CenterX_global_px",
#         y="CenterY_global_px",
#         color="IGF2",
#         size=size,
#         legend_loc = 'none', legend_fontsize=6,
#         legend_fontoutline=2,
#         frameon = False, color_map = "plasma", vmax = 4.0, vmin = 1)

# scv.pl.scatter(
#         adata1,
#         x="CenterX_global_px",
#         y="CenterY_global_px",
#         color="VEGFA",
#         size=size,
#         legend_loc = 'none', legend_fontsize=6,
#         legend_fontoutline=2,
#         frameon = False, color_map = "plasma")

# In[20]:


sc.pl.umap(adata_merge, color = "leiden", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False,  palette = "tab20")


# In[21]:


sc.pl.scatter(
        adata0,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, color_map = "viridis_r")


# In[22]:


sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False)


# In[23]:


sc.pl.scatter(
        adata2,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, color_map = "viridis_r")


# In[24]:


for i in adata1.obs["leiden"].unique():
    sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, groups = i, title = i)


# In[25]:


sc.tl.rank_genes_groups(adata_merge, groupby='leiden', method='wilcoxon', pts = True)
sc.pl.rank_genes_groups(adata_merge, n_genes=25, sharey=False)


# In[26]:


leidenCheck = np.unique(adata_merge.obs["leiden"])

for i in leidenCheck:
    df = sc.get.rank_genes_groups_df(adata_merge, group = i)
    #print(df)
    fileName = "/home/levinsj/Fetal_dir/DEG/Fetal_celltype_neighborhood_ligand_"+str(i)+".csv"
    #print(fileName)
    df.to_csv(fileName)

