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

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 450


# Set parameters

# In[2]:


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


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")


# In[4]:


gene_list = ["DAB2", "IGF2","WT1", "NPHS2"]

filtered_gene_list = [gene for gene in gene_list if gene in adata.var_names]

print(filtered_gene_list)


# In[5]:


sc.tl.score_genes(adata, gene_list=filtered_gene_list, score_name = "IRS_time_related")
adata_1 = adata[adata.obs["sample"] == "1"]


# In[6]:


for i in filtered_gene_list:
    sc.pl.scatter(
        adata_1,
        x="CenterX_global_px",
        y="CenterY_global_px",
        color=i,
        size=5,
        legend_loc = 'none', legend_fontsize=6,
        legend_fontoutline=2,
        frameon = False, layers = "SCVI_imputed", color_map = "viridis_r")


# plt.scatter(adata.obs["PseudoTime_SCVI"], adata.obs["Podo_absorbtion_SCVI"])
# plt.show()

# adata = adata[adata.obs["Podo_absorbtion_SCVI"] > 0.02]
#     
# plt.scatter(adata.obs["PseudoTime_SCVI"], adata.obs["Podo_absorbtion_SCVI"])
# plt.show()

# adata = adata[adata.obs["Podo_absorbtion_SCVI"] > 0.03]
# adata = adata[adata.obs["Podo_absorbtion_SCVI"] - adata.obs["PseudoTime_SCVI"] >  -0.1]
# adata.obs["PseudoTime_SCVI_dec"] = pd.qcut(adata.obs['PseudoTime_SCVI'], q=10, labels=False)
# 
# 
# plt.scatter(adata.obs["PseudoTime_SCVI"], adata.obs["Podo_absorbtion_SCVI"])
# plt.show()

# In[ ]:





# sc.pl.umap(adata, color = "PseudoTime_SCVI_dec", cmap = "plasma", frameon = False)
# 

# sc.pl.umap(adata,color=['SIX2'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['SIX2'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 2)

# sc.pl.umap(adata,color=['WT1'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['WT1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 2)
# sc.pl.umap(adata,color=['WT1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 5)
# sc.pl.umap(adata,color=['WT1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 10)
# sc.pl.umap(adata,color=['WT1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 20)

# sc.pl.umap(adata,color=['CITED1'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['CITED1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 2)
# sc.pl.umap(adata,color=['CITED1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 5)
# sc.pl.umap(adata,color=['CITED1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 10)
# sc.pl.umap(adata,color=['CITED1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 20)

# sc.pl.umap(adata,color=['NPHS2'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['NPHS2'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 2)
# sc.pl.umap(adata,color=['NPHS2'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 5)
# sc.pl.umap(adata,color=['NPHS2'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 10)
# sc.pl.umap(adata,color=['NPHS2'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 20)

# sc.pl.umap(adata,color=['JAG1'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['JAG1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 2)
# sc.pl.umap(adata,color=['JAG1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 5)
# sc.pl.umap(adata,color=['JAG1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 10)
# sc.pl.umap(adata,color=['JAG1'], frameon = False, layer = "counts", cmap = "viridis_r", vmax = 20)

# sc.pl.umap(adata,color=['DAB2'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['SIX1'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['IRX1'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# sc.pl.umap(adata,color=['UNCX'], frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
# 

# print(adata.obs["cellType_SCANVI"].value_counts())
# print(adata.obs["cellType_CosMx_1"].value_counts())

# GOI = ["JAG1","APOE","NPHS2","WT1","UMOD","AQP2","GATA3","KRT19","COL1A1","PLVAP","CD74"]
# order = ["Int","PT","Podocyte","PEC","LOH","DCT","UB_CT","Ureth","Stroma","Endothelium","Immune Cells"]
# 
# sc.pl.dotplot(adata, GOI, groupby='cellType_SCANVI',categories_order = order, log = True, dendrogram=False, cmap='Blues', layer = 'counts')
# sc.pl.dotplot(adata, GOI, groupby='cellType_SCANVI',categories_order = order, log = False, dendrogram=False, cmap='Blues', layer = 'counts')
# 

# order = ["Int","PT","Podocyte","PEC","LOH","DCT","UB_CT","Ureth","Stroma","Endothelium","Immune Cells"]
# 
# sc.pl.dotplot(adata, GOI, groupby='cellType_CosMx_1',categories_order = order, log = True, dendrogram=False, cmap='Blues', layer = 'counts')
# sc.pl.dotplot(adata, GOI, groupby='cellType_CosMx_1',categories_order = order, log = True, dendrogram=False, cmap='Blues', layer = 'counts')
# 

# GOI = ["LHX1","JAG1","CITED1","SIX2","CUBN","SLC3A1","NPHS2","PTPRO","CFH","IRX2","UMOD","WNK1","AQP2","RET","KRT7","DHRS2","PDGFRA","COL1A1","EGFL7","CDH5","HLA-DRA","CD86"]
# order = ["Int","PT","Podocyte","PEC","LOH","DCT","UB_CT","Ureth","Stroma","Endothelium","Immune Cells"]
# 
# sc.pl.dotplot(adata, GOI, groupby='cellType_SCANVI',categories_order = order, log = True, dendrogram=False, cmap='Blues', layer = 'SCVI_imputed')
# sc.pl.dotplot(adata, GOI, groupby='cellType_SCANVI',categories_order = order, log = False, dendrogram=False, cmap='Blues', layer = 'SCVI_imputed')
# 

# order = ["Int","PT","Podocyte","PEC","LOH","DCT","UB_CT","Ureth","Stroma","Endothelium","Immune Cells"]
# 
# sc.pl.dotplot(adata, GOI, groupby='cellType_CosMx_1',categories_order = order, log = True, dendrogram=False, cmap='Blues', layer = 'SCVI_imputed')
# sc.pl.dotplot(adata, GOI, groupby='cellType_CosMx_1',categories_order = order, log = False, dendrogram=False, cmap='Blues', layer = 'SCVI_imputed')
# 

# sc.pl.umap(adata, color = ["cellType_SCANVI"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False, palette = colors)

# fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 9))
# 
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "Podocyte",frameon = False, legend_loc = False, title = "Podocyte", show = False, ax = axes[0,0])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "PEC",frameon = False, legend_loc = False, title = "PEC", show = False, ax = axes[0,1])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "Endothelium",frameon = False, legend_loc = False, title = "Endothelium", show = False, ax = axes[0,2])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "PT",frameon = False, legend_loc = False, title = "PT", show = False, ax = axes[1,0])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "LOH",frameon = False, legend_loc = False, title = "LOH", show = False, ax = axes[1,1])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "DCT",frameon = False, legend_loc = False, title = "DCT", show = False, ax = axes[1,2])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "Stroma",frameon = False, legend_loc = False, title = "Stroma", show = False, ax = axes[2,0])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "UB_CT",frameon = False, legend_loc = False, title = "UB_CT", show = False, ax = axes[2,1])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "Ureth",frameon = False, legend_loc = False, title = "Ureth", show = False, ax = axes[2,2])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "Int",frameon = False, legend_loc = False, title = "Int", show = False, ax = axes[3,0])
# sc.pl.umap(adata, color = "cellType_SCANVI", groups = "Immune Cells",frameon = False, legend_loc = False, title = "Immune Cells", show = False, ax = axes[3,1])
# plt.axis('off')
# plt.show()

# adata.X = adata.layers["counts"]
# adata_stroma = adata[adata.obs["cellType_SCANVI"] == "Stroma"]
# 
# sc.pp.normalize_total(adata_stroma, target_sum=1e4)
# sc.pp.log1p(adata_stroma)
# 
# sc.tl.rank_genes_groups(adata_stroma, 'leiden_sub18', method='wilcoxon')
# sc.pl.rank_genes_groups(adata_stroma, n_genes=25, sharey=False)
# 

# from scipy.sparse import issparse
# from sklearn.preprocessing import MinMaxScaler

# # Assuming 'adata' is your AnnData object
# markers = ["MEIS2","IGF2","COL9A3",
#            "PDGFRA","WNT5A", "PPIA",
#            "POSTN","NR2F1","COL3A1","COL1A1","COL9A3",
#            "REN","ACTA2",
#            "COL4A1","COL4A2","PDGFRB"]
# 
# # Ensure markers list only includes valid gene names
# markers = [gene for gene in markers if gene in adata_stroma.var_names]
# 
# # Convert .X to a dataframe for GOI
# gene_expression_df = pd.DataFrame(
#     adata_stroma[:, markers].X.toarray() if issparse(adata_stroma.X) else adata_stroma[:, markers].X,
#     index=adata_stroma.obs_names,
#     columns=markers
# )
# 
# gene_expression_df['neighborhood'] = adata_stroma.obs['leiden_sub18'].values
# mean_expression_per_cluster = gene_expression_df.groupby('neighborhood').mean()
# 
# scaler = MinMaxScaler()
# gene_expression_scaled = scaler.fit_transform(mean_expression_per_cluster)
# gene_expression_scaled_df = pd.DataFrame(gene_expression_scaled, index=mean_expression_per_cluster.index, columns=mean_expression_per_cluster.columns)
# 
# # Define the order
# order = ["3","1","10","14","18,0","18,1"]
# 
# # Reindex the DataFrame according to the specified order
# grouped_expression = gene_expression_scaled_df.reindex(order)
# 
# # Create the heatmap
# plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
# sns.heatmap(grouped_expression, cmap='plasma')
# # Show the plot
# plt.show()

# adata.X = adata.layers["SCVI_imputed"]
# adata_stroma = adata[adata.obs["cellType_SCANVI"] == "Stroma"]
# 
# sc.pp.normalize_total(adata_stroma, target_sum=1e4)
# sc.pp.log1p(adata_stroma)
# 
# sc.tl.rank_genes_groups(adata_stroma, 'leiden_sub18', method='wilcoxon')
# sc.pl.rank_genes_groups(adata_stroma, n_genes=25, sharey=False)
# 

# # Assuming 'adata' is your AnnData object
# markers = ["FOXD1","IGF2", "MEIS2","NR2F1","FST","COL9A3",
#            "PDGFRA","WNT5A", "PPIA",
#            "POSTN","COL3A1","COL1A1","COL9A3","SPARC",
#            "TBX2","REN","ACTA2",
#            "COL4A1","COL4A2","PDGFRB"]
# 
# # Ensure markers list only includes valid gene names
# markers = [gene for gene in markers if gene in adata_stroma.var_names]
# 
# # Convert .X to a dataframe for GOI
# gene_expression_df = pd.DataFrame(
#     adata_stroma[:, markers].X.toarray() if issparse(adata_stroma.X) else adata_stroma[:, markers].X,
#     index=adata_stroma.obs_names,
#     columns=markers
# )
# 
# gene_expression_df['neighborhood'] = adata_stroma.obs['leiden_sub18'].values
# mean_expression_per_cluster = gene_expression_df.groupby('neighborhood').mean()
# 
# scaler = MinMaxScaler()
# gene_expression_scaled = scaler.fit_transform(mean_expression_per_cluster)
# gene_expression_scaled_df = pd.DataFrame(gene_expression_scaled, index=mean_expression_per_cluster.index, columns=mean_expression_per_cluster.columns)
# 
# # Define the order
# order = ["3","1","10","14","18,0","18,1"]
# 
# # Reindex the DataFrame according to the specified order
# grouped_expression = gene_expression_scaled_df.reindex(order)
# 
# # Create the heatmap
# plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
# sns.heatmap(grouped_expression, cmap='plasma')
# # Show the plot
# plt.show()

# In[7]:


adata.X = adata.layers["SCVI_imputed"]
adata_stroma = adata[adata.obs["cellType_SCANVI"] == "Stroma"]


# In[8]:


neighborhoods = {"1":"1",
                 "2":"2",
                 "3":"3",
                 "4":"4",
                 "5":"5",
                 "6":"6",
                 "7":"7",
                 "8":"8",
                 "9":"9",
                 "11":"11",
                 "12":"12",
                 "13":"13",
                 "15":"15",
                 "16":"16",
                 "17":"17",
                 "19":"19",
                 "20":"20",
                 "21":"21",
                 "22":"22",
                 "23":"23",
                 "24":"24",
                 "25":"25",
                 "10":"10",
                 "14": "14", 
                 "18,1": "18,1", 
                 "18,0": "18,0"}

adata_cosMx = adata_stroma[adata_stroma.obs["tech"] == "CosMx"]
adata_cosMx.obs["leiden_sub18"] = adata_cosMx.obs["leiden_sub18"].map(neighborhoods).astype('str')
keep_values = ["1","3","10","14","18,1" ,"18,0"]
adata_cosMx.obs["leiden_sub18"] = adata_cosMx.obs["leiden_sub18"].where(adata_cosMx.obs["leiden_sub18"].isin(keep_values), "other")
print(adata_cosMx.obs["leiden_sub18"].value_counts())



# In[9]:


neighbor_colors = {
    "1": (234/255, 255/255, 0/255), # yellow
    "3": (3/255, 244/255, 255/255), # blue
    "10": (0/255, 0/255, 0/255), # white
    "14": (238/255, 87/255, 255/255), # purple 
    "18,0" : (40/255, 209/255, 49/255), # green
    "18,1": (255/255, 110/255, 118/255), # bright red
}

# Assuming 'adata_subset' is your AnnData object with the annotation

#cell_type_order = adata.obs["leiden_sub18"].cat.categories.tolist()
cell_type_order = ["1","3",'10','14','18,0','18,1']
# Assuming 'adata_subset' is your AnnData object with the annotation

print(cell_type_order)
colors_mapped = [neighbor_colors[neighborhoods] for neighborhoods in cell_type_order]
adata_cosMx.uns['leiden_sub18_colors'] = colors_mapped

sc.pl.umap(adata_cosMx, color = "leiden_sub18", legend_fontsize=10, legend_fontoutline=2, size = 2, frameon = False)


# In[10]:


print(adata_cosMx.obs["sample"].unique())

