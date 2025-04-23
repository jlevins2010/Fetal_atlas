#!/usr/bin/env python
# coding: utf-8

# Currently running with scanpy_version2.sif

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import matplotlib as mpl


# In[2]:


sc.settings.verbosity = 3
sc.logging.print_header()
mpl.rcParams['figure.dpi'] = 450


# In[3]:


genes_by_counts_max = 2500 # maximum number of counts per cell
genes_by_counts_min = 500 # minimum number of counts per cell
mtThresh_fetal = 25 # percent mitochondrial gene maximum
mtThresh_adult = 40 # percent mitochondrial gene maximum
nPCs_fetal = 40
nPCs_adult = 40
minGenes = 200 # use only cells with at least 200 genes
minCells = 3 # use only genes expressed in at last 3 cells
neighbors = 15

#varsToRegress = ['total_counts', 'pct_counts_mt']
varsToRegress = ['total_counts', 'pct_counts_mt', 'S_score', 'G2M_score']

#cell_cycle_genes = [x.strip() for x in open("/content/drive/MyDrive/SusztakLabFiles/cellCycleGenes.txt")]
cell_cycle_genes = [x.strip() for x in open("/home/levinsj/Fetal_dir/Analysis/referenceFiles/cellCycleGenes.txt")]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]


# In[4]:


adata_merge = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/03_merged/fetal_merged_noAnno_final.h5ad")


# In[5]:


adata_merge.obs.index[0]


# In[6]:


adata_merge.layers


# sc.pl.violin(adata_merge, keys = 'pct_counts_mt', size = 0, groupby = 'sample', rotation= 45)
# sc.pl.violin(adata_merge, keys = 'total_counts', size = 0, groupby = 'sample', rotation= 45)
# sc.pl.violin(adata_merge, keys = 'n_genes_by_counts', size = 0, groupby = 'sample', rotation= 45)
# 

# In[7]:


markers = ['XIST', 'UTY']
sc.pl.stacked_violin(adata_merge, markers, groupby='sample', dendrogram=False, layer = "log1p")
sc.pl.stacked_violin(adata_merge, markers, groupby='sample', dendrogram=False, layer = "counts")


# In[8]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

sc.pl.umap(adata_merge, color = "total_counts", legend_loc= False,frameon = False, show = False, ax = axes[0,0])
sc.pl.umap(adata_merge, color = "pct_counts_mt", legend_loc= False,frameon = False, show = False, ax = axes[0,1])
sc.pl.umap(adata_merge, color = "n_genes_by_counts", legend_loc= False,frameon = False, show = False, ax = axes[0,2])
sc.pl.umap(adata_merge, color = "phase",frameon = False, show = False, ax = axes[1,0])
sc.pl.umap(adata_merge, color = "sample",frameon = False, show = False, ax = axes[1,1])

plt.axis('off')
plt.show()

sc.pl.umap(adata_merge, color = "leiden", legend_loc='on data', frameon = False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')


# In[9]:


adata_merge.X = adata_merge.layers["counts"] # set X-layer prior to subset
adata_merge = adata_merge[~adata_merge.obs.leiden.isin(["18","20","22","10"])]


# In[10]:


sc.pp.normalize_total(adata_merge, target_sum=1e4)
sc.pp.log1p(adata_merge)
adata_merge.layers["log1p"] = adata_merge.X.copy()
sc.pp.neighbors(adata_merge, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
sc.tl.leiden(adata_merge)
sc.tl.paga(adata_merge)
sc.pl.paga(adata_merge, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata_merge, min_dist=0.3)


# In[11]:


sc.pl.umap(adata_merge, color = "leiden", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, palette='Set3')


# In[12]:


sc.tl.rank_genes_groups(adata_merge, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_merge, n_genes=25, sharey=False)


# In[13]:


sc.pl.umap(adata_merge, color = "leiden", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, palette='Set3')


# In[14]:


adata_merge.X = adata_merge.layers["counts"] # set X-layer prior to subset
adata_merge = adata_merge[~adata_merge.obs.leiden.isin(["15","19"])]


# In[15]:


adata_merge.obs["cellType"] = adata_merge.obs["leiden"]
adata_merge.obs["leiden2"] = adata_merge.obs["leiden"]
adata_merge.obs["cellType2"] = adata_merge.obs["leiden"]

cell_identities = {'0': 'Podocyte', '1': 'Stroma', '2': 'NPC', '3': 'Int', '4': 'Int', '5': 'Endothelium', '6': 'Stroma', '7': 'PT', '8': 'DCT', '9': 'LOH', '10': 'Stroma', '11': 'PEC', '12': 'UB_CT', '13': 'Immune Cells', '14': 'Immune Cells', '16': 'Urethelium', '17': 'Endothelium', '18': 'Endothelium', '20': "Stroma"}
adata_merge.obs["cellType"] = adata_merge.obs['leiden'].map(cell_identities).astype('category')

cell_identities = {'0': 'Nephron', '1': 'Stroma', '2': 'Nephron', '3': 'Nephron', '4': 'Nephron', '5': 'Endothelium', '6': 'Stroma', '7': 'Nephron', '8': 'Nephron', '9': 'Nephron', '10': 'Stroma', '11': 'Nephron', '12': 'UB_CT', '13': 'Immune Cells', '14': 'Immune Cells', '16': 'Urethelium', '17': 'Endothelium', '18': 'Endothelium', '20': "Stroma"}
adata_merge.obs["cellType2"] = adata_merge.obs['leiden'].map(cell_identities).astype('category')

cell_identities = {'0': 'Podocyte', '1': 'Stroma', '2': 'NPC', '3': 'Int_1', '4': 'Int_2', '5': 'Endothelium', '6': 'Stroma', '7': 'PT', '8': 'DCT', '9': 'LOH', '10': 'Stroma', '11': 'PEC', '12': 'UB_CT', '13': 'Immune Cells', '14': 'Immune Cells', '16': 'Urethelium', '17': 'Endothelium', '18': 'Endothelium', '20': "Stroma"}
adata_merge.obs["cellType3"] = adata_merge.obs['leiden'].map(cell_identities).astype('category')


# In[16]:


cell_colors = {"NPC": "lavender",
               "Int": "dodgerblue",
               "Podocyte": "teal",
               "PT": "turquoise",
               "DCT": "purple",
               "LOH": "indigo",
               "PEC": "magenta",
               "UB_CT": "thistle",
               "Stroma": "orange",
               "Immune Cells": "palegreen",
               "Endothelium": "dimgrey",
               "Urethelium": "pink",
               "RBC": "gold",
              }

adata_merge.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")

print(adata_merge.layers)

sc.pl.umap(adata_merge, color = ["UMOD"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge, color = ["AQP2"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge, color = ["ATP6V1G3"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge, color = ["cellType"], legend_loc='on data', frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3') 

adata_merge = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")

sc.pl.umap(adata_merge, color = ["UMOD"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge, color = ["AQP2"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge, color = ["ATP6V1G3"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge, color = ["cellType"], legend_loc='on data', frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3') 
sc.pl.umap(adata_merge, color = ["cellType"], legend_loc='none', frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3') 

sc.pl.umap(adata_merge, color = ["cellType"], frameon=False, palette='Set3') 


# In[17]:


sc.pl.umap(adata_merge, color = ["IGF1"], color_map = 'viridis_r', layer = 'counts', frameon = False)
sc.pl.umap(adata_merge, color = ["IGF2"], color_map = 'viridis_r', layer = 'counts', frameon = False)
sc.pl.umap(adata_merge, color = ["IGF1R"], color_map = 'viridis_r', layer = 'counts', frameon = False)
sc.pl.umap(adata_merge, color = ["IGF2R"], color_map = 'viridis_r', layer = 'counts', frameon = False)


# In[18]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
sc.pl.umap(adata_merge, color = "total_counts", legend_loc= False,frameon = False, show = False, ax = axes[0,0])
sc.pl.umap(adata_merge, color = "pct_counts_mt", legend_loc= False,frameon = False, show = False, ax = axes[0,1])
sc.pl.umap(adata_merge, color = "n_genes_by_counts", legend_loc= False,frameon = False, show = False, ax = axes[0,2])
sc.pl.umap(adata_merge, color = "phase",frameon = False, show = False, ax = axes[1,0])
sc.pl.umap(adata_merge, color = "sample",frameon = False, show = False, ax = axes[1,1])
plt.axis('off')
plt.show()


# In[19]:


sc.pl.umap(adata_merge, color = "sample",frameon = False)


# In[20]:


sc.pl.umap(adata_merge, color = ["CITED1","LHX1","RET"], color_map = 'viridis_r', layer = 'counts', frameon = False) 
sc.pl.umap(adata_merge, color = ["CUBN","WNK1","UMOD"], color_map = 'viridis_r', layer = 'counts', frameon = False) 
sc.pl.umap(adata_merge, color = ["NPHS2","CFH","AQP2"], color_map = 'viridis_r', layer = 'counts', frameon = False) 


# In[21]:


GOI = ["LHX1","JAG1","CITED1","SIX2","CUBN","SLC3A1","NPHS2","PTPRO","CFH","IRX2","UMOD","WNK1","AQP2","RET","KRT7","DHRS2","PDGFRA","COL1A1","EGFL7","CDH5","HLA-DRA","CD86"]
order = ["Int","NPC","PT","Podocyte","PEC","LOH","DCT","UB_CT","Urethelium","Stroma","Endothelium","Immune Cells"]

sc.pl.dotplot(adata_merge, GOI, groupby='cellType',categories_order = order, log = True, dendrogram=False, cmap='Blues', layer = 'counts')


# In[22]:


adata_merge


# In[23]:


adata_merge = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")

adata_merge_nephrogenesis = adata_merge[adata_merge.obs['cellType2'].isin(["Nephron","UB_CT","Urethelium"])]


# In[24]:


sc.pl.umap(adata_merge_nephrogenesis, color='cellType', legend_loc='on data',
           frameon=False, legend_fontsize=10, legend_fontoutline=2)

sc.pp.normalize_total(adata_merge_nephrogenesis, target_sum=1e4)
sc.pp.log1p(adata_merge_nephrogenesis)

sc.pp.highly_variable_genes(adata_merge_nephrogenesis, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
sc.pp.regress_out(adata_merge_nephrogenesis, keys = varsToRegress)
sc.pp.neighbors(adata_merge_nephrogenesis, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
sc.tl.leiden(adata_merge_nephrogenesis)
sc.tl.paga(adata_merge_nephrogenesis)
sc.tl.umap(adata_merge_nephrogenesis, min_dist=0.3)
sc.pl.umap(adata_merge_nephrogenesis, color='cellType', legend_loc='on data',
           frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
sc.pl.umap(adata_merge_nephrogenesis, color='leiden', legend_loc='on data',
           frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')

cell_identities = {'0': 'Podocyte', '1': 'NPC', '2': 'Int', '3': 'Int', '4': 'Podocyte', '5': 'PT', '6': 'DCT', '7': 'LOH', '8': 'Int', '9': 'PEC', '10': 'Int', "11": "UB_CT",'12': 'NPC','13': 'Urethelium'}
adata_merge_nephrogenesis.obs["cellType"] = adata_merge_nephrogenesis.obs['leiden'].map(cell_identities).astype('category')

sc.pl.umap(adata_merge_nephrogenesis, color = ["UMOD"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge_nephrogenesis, color = ["AQP2"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge_nephrogenesis, color = ["ATP6V1G3"], color_map = 'viridis_r', layer = 'counts') 
sc.pl.umap(adata_merge_nephrogenesis, color = ["cellType"], legend_loc='on data', frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3') 

sc.pl.umap(adata_merge_nephrogenesis, color='leiden', legend_loc='on data',
           frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')

markers = ["CITED1", "CUBN","PTPRO","CFH", "UMOD","TMEM52B","AQP2","DHRS2"]
order = ["Int","NPC","PT","Podocyte", "PEC", "LOH","DCT","UB_CT", "Urethelium"]
sc.pl.dotplot(adata_merge_nephrogenesis, markers, groupby='cellType',categories_order = order, cmap='Blues', log = True, vmax = 2.5)

sc.pl.dotplot(adata_merge_nephrogenesis, markers, groupby='cellType',dendrogram=True, cmap='Blues', log = False)
adata_merge_nephrogenesis.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalNephronAndUB_annotated_final.h5ad")


# In[25]:


adata_merge.X = adata_merge.layers["counts"]
sc.pp.normalize_total(adata_merge, target_sum=1e4)
sc.pp.log1p(adata_merge)

sc.tl.rank_genes_groups(adata_merge, groupby='cellType', method='wilcoxon', pts = True)
sc.pl.rank_genes_groups(adata_merge, n_genes=25, sharey=False)
leidenCheck = np.unique(adata_merge.obs["cellType"])

for i in leidenCheck:
  df = sc.get.rank_genes_groups_df(adata_merge, group = i)
  #print(df)
  fileName = "/home/levinsj/Fetal_dir/DEG/Fetal_celltype_DEG_"+str(i)+".csv"
  #print(fileName)
  df.to_csv(fileName)


# In[26]:


df = adata_merge.obs["cellType"].value_counts().to_frame()
df.index = adata_merge.obs["cellType"].value_counts().index
df = df.reindex(index = ["NPC","Int","Podocyte","PEC","PT","LOH","DCT","Stroma","Endothelium","UB_CT","Urethelium","Immune Cells"])

print(df)


# In[27]:


df['sampleBreakdown'] = object
df['sample'] = object
df['composition'] = object

### each column is cell Type, parsed by Sample
#gAGE = 12w4d, 14w0d, 15w5d,19w3d,20w5d 
samples = ["HK2718","HK2722","HK2716","HK2725","HK2723"]
cellTypes = ["NPC","Int","Podocyte","PEC","PT","LOH","DCT","Stroma","Endothelium","UB_CT","Urethelium","Immune Cells"]

for i in df.index:
    counts_sample = []
    for j in samples:
        counts_sample.append(adata_merge.obs[(adata_merge.obs['sample'] == j) & (adata_merge.obs['cellType'] == i)].shape[0])
    df.at[i,'sampleBreakdown'] = counts_sample

print(df)    


# In[28]:


# Set the width of each bar and the space between the bars Orange is Healthy
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


# adata_merge = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")
# sc.pl.umap(adata_merge, color = ["UMOD"], color_map = 'viridis_r', layer = 'counts') 
# 
# adata_merge_stroma = adata_merge[adata_merge.obs['cellType'].isin(["Stroma"])]
# 
# sc.pl.umap(adata_merge_stroma, color='cellType', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2)
# 
# sc.pp.normalize_total(adata_merge_stroma, target_sum=1e4)
# sc.pp.log1p(adata_merge_stroma)
# sc.pp.highly_variable_genes(adata_merge_stroma, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
# sc.pp.regress_out(adata_merge_stroma, keys = varsToRegress)
# sc.pp.neighbors(adata_merge_stroma, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
# sc.tl.leiden(adata_merge_stroma)
# sc.tl.paga(adata_merge_stroma)
# sc.tl.umap(adata_merge_stroma, min_dist=0.3)
# sc.pl.umap(adata_merge_stroma, color='cellType', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
# sc.pl.umap(adata_merge_stroma, color='leiden', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
# 
# sc.tl.rank_genes_groups(adata_merge_stroma, 'leiden', method='wilcoxon')
# sc.pl.rank_genes_groups(adata_merge_stroma, n_genes=25, sharey=False)
# 

# adata_merge_stroma.X = adata_merge_stroma.layers["counts"] # X-layer set prior to subsetting
# adata_merge_stroma = adata_merge_stroma[~adata_merge_stroma.obs.leiden.isin(["10","11"])]
# sc.pp.normalize_total(adata_merge_stroma, target_sum=1e4)
# sc.pp.log1p(adata_merge_stroma)
# sc.pp.highly_variable_genes(adata_merge_stroma, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
# sc.pp.regress_out(adata_merge_stroma, keys = varsToRegress)
# sc.pp.neighbors(adata_merge_stroma, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
# sc.tl.leiden(adata_merge_stroma)
# sc.tl.paga(adata_merge_stroma)
# sc.tl.umap(adata_merge_stroma, min_dist=0.3)
# 
# sc.pl.umap(adata_merge_stroma, color='cellType', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
# sc.pl.umap(adata_merge_stroma, color='leiden', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
# sc.pl.umap(adata_merge_stroma, color='sample',frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')

# sc.tl.rank_genes_groups(adata_merge_stroma, 'leiden', method='wilcoxon')
# sc.pl.rank_genes_groups(adata_merge_stroma, n_genes=25, sharey=False)

# adata_merge_stroma.obs["time"] = adata_merge_stroma.obs["sample"]
# age = ["15.5", "12.4", "14.0", "20.5", "19.3"]
# adata_merge_stroma.rename_categories('time', age)
# 
# sc.pl.umap(adata_merge_stroma[adata_merge_stroma.obs["time"].isin(["12.4"])], color="leiden", show=False, title = "12.4")
# sc.pl.umap(adata_merge_stroma[adata_merge_stroma.obs["time"].isin(["14.0"])], color="leiden", show=False, title = "14.0")
# sc.pl.umap(adata_merge_stroma[adata_merge_stroma.obs["time"].isin(["15.5"])], color="leiden", show=False, title = "15.5")
# sc.pl.umap(adata_merge_stroma[adata_merge_stroma.obs["time"].isin(["19.3"])], color="leiden", show=False, title = "19.3")
# sc.pl.umap(adata_merge_stroma[adata_merge_stroma.obs["time"].isin(["20.5"])], color="leiden", show=False, title = "20.5")

# sc.pl.umap(adata_merge_stroma, color = ["COL1A1"], color_map = 'viridis_r', layer = 'counts') 
# 
# adata_merge_stroma.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalStromaOnly_annotated_final.h5ad")

# adata_merge = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")
# 
# adata_merge_end = adata_merge[adata_merge.obs['cellType'].isin(["Endothelium"])]
# sc.pl.umap(adata_merge_end, color='cellType', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2)
# 
# sc.pp.normalize_total(adata_merge_end, target_sum=1e4)
# sc.pp.log1p(adata_merge_end)
# sc.pp.highly_variable_genes(adata_merge_end, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
# sc.pp.regress_out(adata_merge_end, keys = varsToRegress)
# sc.pp.neighbors(adata_merge_end, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
# sc.tl.leiden(adata_merge_end)
# sc.tl.paga(adata_merge_end)
# sc.tl.umap(adata_merge_end, min_dist=0.3)
# sc.pl.umap(adata_merge_end, color='cellType', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
# sc.pl.umap(adata_merge_end, color='leiden', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')

# adata_merge_end.X = adata_merge_end.layers["counts"] # X-layer set prior to subsetting
# # remove cluster 6 (podocytes) and cluster 11, RBCs
# adata_merge_end = adata_merge_end[~adata_merge_end.obs.leiden.isin(["8","10"])]
# 
# sc.pp.normalize_total(adata_merge_end, target_sum=1e4)
# sc.pp.log1p(adata_merge_end)
# sc.pp.highly_variable_genes(adata_merge_end, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
# sc.pp.regress_out(adata_merge_end, keys = varsToRegress)
# sc.pp.neighbors(adata_merge_end, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
# sc.tl.leiden(adata_merge_end)
# sc.tl.paga(adata_merge_end)
# sc.tl.umap(adata_merge_end, min_dist=0.3)
# sc.pl.umap(adata_merge_end, color='cellType', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
# sc.pl.umap(adata_merge_end, color='leiden', legend_loc='on data',
#            frameon=False, legend_fontsize=10, legend_fontoutline=2, palette='Set3')
# 
# sc.pl.umap(adata_merge_end, color = ["IGF1"], color_map = 'viridis_r') 
# sc.pl.umap(adata_merge_end, color = ["GJA4"], color_map = 'viridis_r') 
# 
# sc.pl.umap(adata_merge_end, color='EMCN', color_map = 'viridis_r', layer = 'counts')
# 
# adata_merge_end.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalEndotheliumOnly_annotated_final.h5ad")

# sc.tl.rank_genes_groups(adata_merge_end, 'leiden', method='wilcoxon')
# sc.pl.rank_genes_groups(adata_merge_end, n_genes=25, sharey=False)
