#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad

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


# In[3]:


from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler


# Set parameters

# In[4]:


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


# In[5]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression_neighborhoods.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
print(adata)


# In[6]:


adata1 = adata[adata.obs["sample"] == "1"]
print(adata.obs["leiden_neighborhood"].value_counts())


# In[7]:


sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, color_map = "viridis_r", groups = "6")


sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, color_map = "viridis_r", groups = "5")

sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, color_map = "viridis_r", groups = "1")

sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, color_map = "viridis_r", groups = "7")

sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, color_map = "viridis_r", groups = "0")


# In[8]:


adata = adata[adata.obs['cellType_SCANVI'] == "Stroma"]
print(adata)
sc.tl.rank_genes_groups(adata, groupby='leiden_neighborhood', method='wilcoxon', pts = True)
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)


# In[9]:


adata = adata[adata.obs['cellType_SCANVI'] == "Stroma"]
adata_df = adata[adata.obs["leiden_neighborhood"].isin(["6","5","1","0","7"])]

# Assuming 'adata' is your AnnData object
markers = ["FOXD1", "IGF2","WNT4","IGF1", "A2M"]

# Ensure markers list only includes valid gene names
markers = [gene for gene in markers if gene in adata.var_names]

# Convert .X to a dataframe for GOI
gene_expression_df = pd.DataFrame(
    adata_df[:, markers].X.toarray() if issparse(adata.X) else adata_df[:, markers].X,
    index=adata_df.obs_names,
    columns=markers
)

gene_expression_df['leiden_neighborhood'] = adata_df.obs['leiden_neighborhood'].values
mean_expression_per_cluster = gene_expression_df.groupby('leiden_neighborhood').mean()

scaler = MinMaxScaler()
gene_expression_scaled = scaler.fit_transform(mean_expression_per_cluster)
gene_expression_scaled_df = pd.DataFrame(gene_expression_scaled, index=mean_expression_per_cluster.index, columns=mean_expression_per_cluster.columns)

# Define the order
order = ["6","5","1","0","7"]

# Reindex the DataFrame according to the specified order
grouped_expression = gene_expression_scaled_df.reindex(order)

# Create the heatmap
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
sns.heatmap(grouped_expression, cmap='plasma')
# Show the plot
plt.show()

