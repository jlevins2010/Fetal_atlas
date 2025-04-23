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

color2 = {
    "0": "#1F77B4",
    "1": "#FF7F0E",
    "2": "#2CA02C",
    "3": "#D62728",
    "4": "#9467BD",
    "5": "#8C564B",
    "6": "#E377C2",
    "7": "#4f8215",
    "8": "#BCBD22",
    "9": "#17BECF"
}


# In[5]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression_neighborhoods.h5ad")
adata = adata[adata.obs["tech"] == "CosMx"]
print(adata)


# In[6]:


color_palette = [
    "#FF0000",  # Red - 1
    "#FFA500",  # Orange - 2
    "#FFFF00",  # Yellow - 3
    "#008000",  # Green - 4
    "#0000FF",  # Blue - 5
    "#800080",  # Purple - 6
    "#FFC0CB",  # Pink - 7
    "#804000",  # Brown - 8
    "#FFFFFF",  # Gray - 9
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
    "#000000"   # White
]


# In[7]:


sc.pl.umap(adata, color = "leiden_neighborhood", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False, palette = color_palette)


# In[8]:


adata1 = adata[adata.obs["sample"] == "1"]
print(adata.obs["leiden_neighborhood"].value_counts())
print(adata1.obs["leiden_neighborhood"].value_counts())


# In[9]:


sc.pl.umap(adata1, color = "leiden_neighborhood", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False)


# sc.pl.umap(adata, color = "leiden_neighborhood", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False, palette = color_orig)
# 

# In[10]:


sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False)


# In[11]:


sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, groups = "6")


sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, groups = "5")

sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, groups = "1")

sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, groups = "7")

sc.pl.scatter(
        adata1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color="leiden_neighborhood",
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, groups = "0")


# In[12]:


order = ["10","7","0","1","8"]


# In[13]:


adata = adata[adata.obs['NPC_SCVI'] >= 0]
adata_df = adata[adata.obs["leiden_neighborhood"].isin(order)]

# Assuming 'adata' is your AnnData object
markers = ["IGF2", "MEIS2","UNCX","SIX2","SIX1","LHX1","ENO1","JAG1","WT1","NPHS2"]

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

# Reindex the DataFrame according to the specified order
grouped_expression = gene_expression_scaled_df.reindex(order)

# Create the heatmap
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
sns.heatmap(grouped_expression, cmap='plasma')
# Show the plot
plt.show()


# In[14]:


adata = adata[adata.obs['NPC_SCVI'] >= 0]
adata_df = adata[adata.obs["leiden_neighborhood"].isin(order)]

# Assuming 'adata' is your AnnData object
markers = ['PseudoTime_SCVI','NPC_SCVI', 'PT_absorbtion_SCVI', 'DCT_absorbtion_SCVI','LOH_absorbtion_SCVI','Podo_absorbtion_SCVI', 'G2M_score']

# Ensure markers list only includes valid gene names
gene_expression_df = adata_df.obs[markers]

gene_expression_df['leiden_neighborhood'] = adata_df.obs['leiden_neighborhood'].values
mean_expression_per_cluster = gene_expression_df.groupby('leiden_neighborhood').mean()

scaler = MinMaxScaler()
gene_expression_scaled = scaler.fit_transform(mean_expression_per_cluster)
gene_expression_scaled_df = pd.DataFrame(gene_expression_scaled, index=mean_expression_per_cluster.index, columns=mean_expression_per_cluster.columns)

# Define the order
# Reindex the DataFrame according to the specified order
grouped_expression = gene_expression_scaled_df.reindex(order)

# Create the heatmap
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
sns.heatmap(grouped_expression, cmap='plasma')
# Show the plot
plt.show()


# In[15]:


adataPT = adata[adata.obs["cellType_SCANVI"] == "PT"]
sc.pl.violin(adataPT, keys="Glomerular_SCVI", groupby="leiden_neighborhood", order=order)

