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


# Set parameters

# In[2]:


neighbors = 15
nPCs = 30
leiden_resolution = 1


# In[3]:


adata1 = sc.read_h5ad("/home/levinsj/spatial/adata/individual_Samples/HK3524_raw_noImages.h5ad")
adata2 = sc.read_h5ad("/home/levinsj/spatial/adata/individual_Samples/FK1_raw_noImages.h5ad")
adata3 = sc.read_h5ad("/home/levinsj/spatial/adata/individual_Samples/FK4_raw_noImages.h5ad")


# In[4]:


adata1.obs.index[0:19]


# In[5]:


samples = [adata1, adata2, adata3]

adata_merge = samples[0].concatenate(samples[1:],
    join="inner",
    batch_key="sample"
)


# In[6]:


print(adata_merge)


# In[7]:


adata_merge.obs.index[0:19]


# In[8]:


adata_merge.var_names


# remove system controls and negative probes from .X layer

# In[9]:


systemChecks = adata_merge.var_names.str.startswith('SystemControl')
neg_probes = adata_merge.var_names.str.startswith('Negative')

remove = np.add(neg_probes, systemChecks)
keep = np.invert(remove)

adata_merge = adata_merge[:,keep]


# calculate reads per region

# In[10]:


print(adata_merge)


# In[11]:


memCounts = adata_merge.layers["Membrane"].sum(axis=1)
nucCounts = adata_merge.layers["Nuclear"].sum(axis=1)
cytCounts = adata_merge.layers["Cytoplasm"].sum(axis=1)
totalCounts = adata_merge.layers["All_counts"].sum(axis=1)

adata_merge.obs["pct_nuc"] = nucCounts/totalCounts
adata_merge.obs["pct_mem"] = memCounts/totalCounts
adata_merge.obs["pct_cyt"] = cytCounts/totalCounts

sc.pl.violin(adata_merge, keys = ['pct_nuc', 'pct_mem', 'pct_cyt'],  size = 0, groupby = 'sample', rotation= 45)


# In[12]:


sc.pl.violin(adata_merge, keys = ['nCount_RNA', 'nFeature_RNA'],  size = 0, groupby = 'sample', rotation= 45)
sc.pl.violin(adata_merge, keys = ['Area', 'Width', 'Height'],  size = 0, groupby = 'sample', rotation= 45)


# In[13]:


sc.pp.filter_cells(adata_merge, min_counts=30)
sc.pp.filter_genes(adata_merge, min_cells=400)

# remove Malat1, mitochondrial genes and hemoglobin genes
adata_merge.X = adata_merge.layers["All_counts"]
malat1 = adata_merge.var_names.str.startswith('MALAT1')
mito_genes = adata_merge.var_names.str.startswith('MT-')
hb_genes = adata_merge.var_names.str.contains('^HB[^(P)]')

remove = np.add(mito_genes, malat1)
remove = np.add(remove, hb_genes)

keep = np.invert(remove)

adata_merge = adata_merge[:,keep]

print(adata_merge.n_obs, adata_merge.n_vars)


# In[14]:


sc.pp.normalize_total(adata_merge, inplace=True)
sc.pp.log1p(adata_merge)
sc.pp.pca(adata_merge)
sc.pp.neighbors(adata_merge, n_neighbors = neighbors, n_pcs=nPCs)
sc.tl.leiden(adata_merge, resolution = leiden_resolution)
sc.tl.umap(adata_merge, min_dist = 0.01)


# In[15]:


sc.pl.umap(adata_merge, color = "leiden", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[16]:


sc.tl.rank_genes_groups(adata_merge, groupby='leiden', method='wilcoxon', pts = True)
sc.pl.rank_genes_groups(adata_merge, n_genes=25, sharey=False)


# In[17]:


sc.tl.umap(adata_merge, min_dist = 0.01)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 3))

sc.pl.umap(adata_merge, color = "sample", groups = "0",frameon = False, legend_loc = False, title = "HK3524", show = False, ax = axes[0])
sc.pl.umap(adata_merge, color = "sample", groups = "1",frameon = False, legend_loc = False, title = "FK1", show = False, ax = axes[1])
sc.pl.umap(adata_merge, color = "sample", groups = "2",frameon = False, legend_loc = False, title = "FK4", show = False, ax = axes[2])
plt.axis('off')

plt.show()


# In[18]:


adata_merge.obs["leiden"].value_counts()
cell_identities = {'0': 'Stroma_1', '1': 'Intermediate_1', '2': 'Intermediate_2', '3': 'Stroma_2', '4': 'PT', '5': 'Podo', '6': 'Endothelium_1', '7': 'LOH', '8': 'Endothelium_2', '9': 'Immune', '10': 'Stroma_3', '11': 'CD', '12':'UB'}
adata_merge.obs["cellType_CosMx_1"] = adata_merge.obs['leiden'].map(cell_identities).astype('category')

sc.pl.umap(adata_merge, color = "cellType_CosMx_1", frameon = False)
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[19]:


fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))

sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "PT",frameon = False, legend_loc = False, title = "PT", show = False, ax = axes[0,0])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Podo",frameon = False, legend_loc = False, title = "Podo", show = False, ax = axes[0,1])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "LOH",frameon = False, legend_loc = False, title = "LOH", show = False, ax = axes[0,2])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Intermediate_1",frameon = False, legend_loc = False, title = "Intermediate_1", show = False, ax = axes[0,3])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Intermediate_2",frameon = False, legend_loc = False, title = "Intermediate_2", show = False, ax = axes[0,4])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "UB",frameon = False, legend_loc = False, title = "UB", show = False, ax = axes[1,0])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "CD",frameon = False, legend_loc = False, title = "CD", show = False, ax = axes[1,1])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Stroma_1",frameon = False, legend_loc = False, title = "Stroma_1", show = False, ax = axes[1,2])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Stroma_2",frameon = False, legend_loc = False, title = "Stroma_2", show = False, ax = axes[1,3])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Stroma_3",frameon = False, legend_loc = False, title = "Stroma_3", show = False, ax = axes[1,4])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Endothelium_1",frameon = False, legend_loc = False, title = "Endothelium_1", show = False, ax = axes[2,0])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Endothelium_2",frameon = False, legend_loc = False, title = "Endothelium_2", show = False, ax = axes[2,1])
sc.pl.umap(adata_merge, color = "cellType_CosMx_1", groups = "Immune",frameon = False, legend_loc = False, title = "Immune", show = False, ax = axes[2,2])
plt.axis('off')
plt.show()


# In[20]:


colors = {"PT": "#800515",
               "Podo": "#bd5713",
               "LOH": "#63ad2d", 
               "Intermediate_1": "#a4b507",
               "Intermediate_2": "black",
               "UB": "#6e640e", 
               "CB": "#1c768a",
               "Stroma_1": "#3e9c72",
               "Stroma_2": "#448efc",
               "Stroma_3": "#51657a", 
               "Endothelium_1": "#6f37b3",
               "Endothelium_2": "#ff0011",
               "Immune": "#b44dbf",
               "CD": "#4f6e57"
         }


# In[21]:


df = adata_merge.obs["cellType_CosMx_1"].value_counts().to_frame()
df.index = adata_merge.obs["cellType_CosMx_1"].value_counts().index

df['sampleBreakdown'] = object
df['sampleOrder_connectivity'] = object
df['nn1'] = object
df['nn20'] = object
df['nn100'] = object
df['micron20'] = object


print(df.index)


# In[22]:


# Set the width of each bar and the space between the bars
width = 0.9

# Loop over each cell type and create a stacked bar plot of the "species" category
for i, cell_type in enumerate(df.index):
    ind = np.arange(1)
    S1 = df["count"]
    sample_type = df.index[i]
    #p1 = plt.bar(i, S1[i], width, color = colors[sample_type])
    p1 = plt.bar(i, S1[i], width, color = colors[sample_type])
    # Show the plot
plt.tight_layout()

plt.show()


# In[23]:


for i in df.index:
    counts_sample = []
    for j in adata_merge.obs["sample"].unique():
        counts_sample.append(adata_merge.obs[(adata_merge.obs['sample'] == j) & (adata_merge.obs['cellType_CosMx_1'] == i)].shape[0])
    df.at[i,'sampleBreakdown'] = counts_sample

# Set the width of each bar and the space between the bars
width = 0.9

# Loop over each cell type and create a stacked bar plot of the "sample" category
for i, cell_type in enumerate(df.index):
    ind = np.arange(1)
    S1 = df.sampleBreakdown[i]

    palette = plt.cm.get_cmap('tab10')
    bottom = 0
    for j in range(len(S1)):
          plt.bar(i, S1[j],bottom = bottom, width = width, color = palette(j))
          bottom = bottom + S1[j]
    # Show the plot
plt.tight_layout()

plt.show()


# remove more genes

# In[24]:


other_genes = adata_merge.var_names.str.startswith(('UTY','XIST'))
keep = np.invert(other_genes)
adata_merge = adata_merge[:,keep]
print(adata_merge.n_obs, adata_merge.n_vars)


# In[25]:


adata_merge.obs['Count_percentile'] = adata_merge.obs['nCount_RNA'].rank(pct=True) * 100


# In[26]:


adata_merge.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/fetal_annotated_AllCells.h5ad")

