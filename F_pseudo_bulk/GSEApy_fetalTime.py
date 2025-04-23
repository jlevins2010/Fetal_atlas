#!/usr/bin/env python
# coding: utf-8

# Run with /home/levinsj/Fetal_dir/Analysis/sif/pymetaneighbor_version3.sif environment

# In[1]:


import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pymn
import gseapy

from gseapy import barplot, dotplot


# In[2]:


#These save characters as text in PDFs
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#These change plot aesthetics

sns.set(style='white', font_scale=1.25)
plt.rc("axes.spines", top=False, right=False)
plt.rc('xtick', bottom=True)
plt.rc('ytick', left=True)


# In[3]:


human = gseapy.get_library_name(organism='Human')
print(human)


# Plotting of DEGs being conserved

# In[4]:


fetalDT = pd.read_csv("/home/levinsj/Fetal_dir/DEG/DCT_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')
fetalPT = pd.read_csv("/home/levinsj/Fetal_dir/DEG/PT_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')
fetalLOH = pd.read_csv("/home/levinsj/Fetal_dir/DEG/LOH_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')
fetalNPC = pd.read_csv("/home/levinsj/Fetal_dir/DEG/NPC_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')

fetalPodo = pd.read_csv("/home/levinsj/Fetal_dir/DEG/Podocyte_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')
fetalStroma = pd.read_csv("/home/levinsj/Fetal_dir/DEG/Stroma_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')
fetalEndothelial = pd.read_csv("/home/levinsj/Fetal_dir/DEG/Endothelium_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')


fetalPT["names"] = fetalPT.index
fetalLOH["names"] = fetalLOH.index
fetalNPC["names"] = fetalNPC.index
fetalPodo["names"] = fetalPodo.index
fetalStroma["names"] = fetalStroma.index
fetalEndothelial["names"] = fetalEndothelial.index


# In[5]:


populations_to_test = [fetalPT,fetalDT,fetalLOH,fetalNPC,fetalPodo,fetalStroma,fetalEndothelial]

for i in populations_to_test:
    i["names"] = i.index
    i["pi_score"] = -1 * np.log10(i["pvalue"]) * i["log2FoldChange"] # pvalue
    i.dropna(inplace=True)


# In[6]:


v_library='WikiPathway_2023_Human' ## 'WikiPathway_2023_Human' ###  
gset = gseapy.parser.get_library(v_library)
print(gset)


# In[7]:


min_size = 5
max_size = 100


# In[8]:


adata_merge = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")
background = adata_merge.var_names


# # PT genes

# In[9]:


gene_rank = fetalPT[['names','pi_score']]
gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
gene_rank = gene_rank.reset_index(drop=True)
print(gene_rank.head())

res = gseapy.prerank(rnk=gene_rank, gene_sets=gset, 
                    min_size=min_size,
                    max_size=max_size,
                    permutation_num=1000, # reduce number to speed up testing
                    outdir=None, # don't write to disk
                    seed=6,
                    verbose=True,)

res.res2d.sort_values(by=['FDR q-val'], inplace=True, ascending=True)
res.res2d.head(5)

#ax = dotplot(res.res2d, title='WikiPathway 2023 Human',cmap='viridis_r', size=10, figsize=(3,5))
# needs adjusted p-Value for dotplot which is not compatible with pi_score variable name


# In[10]:


print(fetalPT)


# In[11]:


gene_rank = fetalPT[['names','padj','log2FoldChange']]
gene_rank = gene_rank[gene_rank.padj <= 0.05]
gene_rank.sort_values(by=['padj'], inplace=True, ascending=False)
gene_rank = gene_rank.reset_index(drop=True)
print(gene_rank.head())

enr_up = gseapy.enrichr(gene_rank.names[gene_rank.log2FoldChange > 0],
                    gene_sets= gset,
                    background= background,
                    outdir=None)

gseapy.dotplot(enr_up.res2d, figsize=(4,5), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
plt.show()

enr_down = gseapy.enrichr(gene_rank.names[gene_rank.log2FoldChange < 0],
                    gene_sets= gset,
                    background= background,
                    outdir=None)

gseapy.dotplot(enr_down.res2d, figsize=(3,5), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
plt.show()


# ## NPC cells

# In[12]:


# GSEApy analysis


# In[13]:


gene_rank = fetalNPC[['names','pi_score']]
gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
gene_rank = gene_rank.reset_index(drop=True)
print(gene_rank.head())

res = gseapy.prerank(rnk=gene_rank, gene_sets=gset, 
                    min_size=min_size,
                    max_size=max_size,
                    permutation_num=1000, # reduce number to speed up testing
                    outdir=None, # don't write to disk
                    seed=6,
                    verbose=True,)

res.res2d.sort_values(by=['FDR q-val'], inplace=True, ascending=True)
res.res2d.head(5)

#ax = dotplot(res.res2d, title='WikiPathway 2023 Human',cmap='viridis_r', size=10, figsize=(3,5))


# In[14]:


gene_rank = fetalNPC[['names','padj','log2FoldChange']]
gene_rank = gene_rank[gene_rank.padj <= 0.05]
gene_rank.sort_values(by=['padj'], inplace=True, ascending=False)
gene_rank = gene_rank.reset_index(drop=True)
print(gene_rank.head())

enr_up = gseapy.enrichr(gene_rank.names[gene_rank.log2FoldChange > 0],
                    gene_sets= gset,
                    background= background,
                    outdir=None)

gseapy.dotplot(enr_up.res2d, figsize=(5,15), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
plt.show()

enr_down = gseapy.enrichr(gene_rank.names[gene_rank.log2FoldChange < 0],
                    gene_sets= gset,
                    background= background,
                    outdir=None)

gseapy.dotplot(enr_down.res2d, figsize=(5,15), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
plt.show()


# In[15]:


plt.rcParams.update({'font.size': 1})


# In[16]:


gene_rank = fetalNPC[['names','padj','log2FoldChange']]
gene_rank = gene_rank[gene_rank.padj <= 0.05]
gene_rank.sort_values(by=['padj'], inplace=True, ascending=False)
gene_rank = gene_rank.reset_index(drop=True)
print(gene_rank.head())

enr_up = gseapy.enrichr(gene_rank.names[gene_rank.log2FoldChange > 0],
                    gene_sets= gset,
                    background= background,
                    outdir=None)

gseapy.dotplot(enr_up.res2d, figsize=(3,5), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
plt.show()

enr_down = gseapy.enrichr(gene_rank.names[gene_rank.log2FoldChange < 0],
                    gene_sets= gset,
                    background= background,
                    outdir=None)

gseapy.dotplot(enr_down.res2d, figsize=(3,5), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
plt.show()


# In[17]:


## DCT cells


# gene_rank = fetalDT[['names','pi_score']]
# gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
# gene_rank = gene_rank.reset_index(drop=True)
# 
# res = gseapy.prerank(rnk=gene_rank, gene_sets=gset, 
#                     min_size=25,
#                     max_size=200,
#                     permutation_num=1000, # reduce number to speed up testing
#                     outdir=None, # don't write to disk
#                     seed=6,
#                     verbose=True,)
# 
# res.res2d.sort_values(by=['FDR q-val'], inplace=True, ascending=True)
# res.res2d.head(20)

# enr_up = gseapy.enrichr(gene_rank.names[gene_rank.pi_score >= piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_up.res2d, figsize=(3,5), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()
# 
# enr_down = gseapy.enrichr(gene_rank.names[gene_rank.pi_score <= -piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_down.res2d, figsize=(3,5), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()

# ## LOH Cells

# gene_rank = fetalLOH[['names','pi_score']]
# gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
# gene_rank = gene_rank.reset_index(drop=True)
# 
# res = gseapy.prerank(rnk=gene_rank, gene_sets=gset)
# 
# res.res2d.head(20)

# enr_up = gseapy.enrichr(gene_rank.names[gene_rank.pi_score >= piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_up.res2d, figsize=(3,5), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()
# 
# enr_down = gseapy.enrichr(gene_rank.names[gene_rank.pi_score <= -piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_down.res2d, figsize=(3,5), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()

# ## Podocytes

# gene_rank = fetalPodo[['names','pi_score']]
# gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
# gene_rank = gene_rank.reset_index(drop=True)
# 
# res = gseapy.prerank(rnk=gene_rank, gene_sets=gset)
# 
# res.res2d.head(20)

# enr_up = gseapy.enrichr(gene_rank.names[gene_rank.pi_score >= piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_up.res2d, figsize=(3,5), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()
# 
# enr_down = gseapy.enrichr(gene_rank.names[gene_rank.pi_score <= -piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_down.res2d, figsize=(3,5), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()

# ## Stromal cells

# gene_rank = fetalStroma[['names','pi_score']]
# gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
# gene_rank = gene_rank.reset_index(drop=True)
# 
# res = gseapy.prerank(rnk=gene_rank, gene_sets=gset)
# 
# res.res2d.head(20)

# enr_up = gseapy.enrichr(gene_rank.names[gene_rank.pi_score >= piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_up.res2d, figsize=(3,5), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()
# 
# enr_down = gseapy.enrichr(gene_rank.names[gene_rank.pi_score <= -piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_down.res2d, figsize=(3,5), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()

# In[18]:


## Endothelial Cells


# gene_rank = fetalEndothelial[['names','pi_score']]
# gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
# gene_rank = gene_rank.reset_index(drop=True)
# 
# res = gseapy.prerank(rnk=gene_rank, gene_sets=gset)
# 
# res.res2d.head(20)

# enr_up = gseapy.enrichr(gene_rank.names[gene_rank.pi_score >= piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_up.res2d, figsize=(3,5), title="Increase with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()
# 
# enr_down = gseapy.enrichr(gene_rank.names[gene_rank.pi_score <= -piCutoff],
#                     gene_sets= gset,
#                     outdir=None)
# 
# gseapy.dotplot(enr_down.res2d, figsize=(3,5), title="Decrease with Gestational Age", cmap = plt.cm.autumn_r)
# plt.show()
