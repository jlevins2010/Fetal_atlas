#!/usr/bin/env python
# coding: utf-8

# Run with /home/levinsj/Fetal_dir/Analysis/sif/pymetaneighbor_version2.sif environment

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


# Import time associated Genes

# In[3]:


fetalNPC = pd.read_csv("/home/levinsj/Fetal_dir/DEG/NPC_fetal_time_dependent_genes.csv", index_col = 'Unnamed: 0')
fetalNPC["names"] = fetalNPC.index

print(fetalNPC)


# Import CellRank genes

# In[4]:


CellRankDrivers = pd.read_csv("/home/levinsj/Fetal_dir/Velocyto/03_CellRank/NephroLineageDrivers_self_renew.csv", index_col = 'Unnamed: 0')
CellRankDrivers["names"] = CellRankDrivers.index
print(CellRankDrivers)


# In[5]:


merged_df = pd.merge(fetalNPC, CellRankDrivers, how='inner', left_index=True, right_index=True)
print(merged_df)

merged_df["TimeDependent_log10"] = -np.log10(merged_df["padj"])
merged_df["CellRank_log10"] = -np.log10(merged_df["NPC_qval"])


# In[6]:


signif = merged_df.NPC_qval <= 0.05
renew = merged_df.NPC_corr > 0
dif = merged_df.NPC_corr < 0

merged_df['renew'] = np.where((signif & renew), 1, 0)
merged_df['dif'] = np.where((signif & dif), 1, 0)

merged_df.to_csv("/home/levinsj/Fetal_dir/DEG/NPC_fetal_time_dependent_genes_with_Renew.csv")


# In[7]:


plt.figure(figsize=(8,5))
ax = sns.scatterplot(x="log2FoldChange", y="TimeDependent_log10",hue = "renew",  data=merged_df, linewidth=0)

genes_to_label = ["SIX2", "IGF2BP1", "RSPO3", "GFRA1","HIST1H4C","TMEM258"]

filtered_df = merged_df[merged_df.index.isin(genes_to_label)]

for i, row in filtered_df.iterrows():
    plt.annotate(row.name,
                 xy=(row['log2FoldChange'], row['TimeDependent_log10']),
                 xytext=(20, 20),  # Adjust offset as needed
                 textcoords='offset points', fontsize = 20,
                 arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", color = "black"),
                 ha='right', va='center')
    
plt.show()


# In[8]:


plt.figure(figsize=(8,5))
ax = sns.scatterplot(x="log2FoldChange", y="TimeDependent_log10",hue = "dif",  data=merged_df, linewidth=0)

genes_to_label = ["CA7", "MR1", "WNT4", "NACC2", "SEMA5A", "APLP2","FABP4"]

# Filter for the desired indices/genes
filtered_df = merged_df[merged_df.index.isin(genes_to_label)]

# Annotate points with indices/gene names
for i, row in filtered_df.iterrows():
    plt.annotate(row.name,
                 xy=(row['log2FoldChange'], row['TimeDependent_log10']),
                 xytext=(10, 10),  # Adjust offset as needed
                 textcoords='offset points', fontsize = 20,
                 arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.4", color = "black"),
                 ha='left', va='top')
    
plt.show()


# In[9]:


human = gseapy.get_library_name(organism='Human')

v_library='WikiPathway_2023_Human' ## 'WikiPathway_2023_Human' ###  
gset = gseapy.parser.get_library(v_library, min_size=5)


# In[10]:


min_size = 30
max_size = 100

cellRank = merged_df
cellRank["names"]= cellRank["names_x"]

cellRank["pi_score"] = -1 * np.log10(cellRank["NPC_corr"]) * cellRank["NPC_qval"]
cellRank.dropna(inplace=True)

gene_rank = cellRank[['names','pi_score']]
gene_rank.sort_values(by=['pi_score'], inplace=True, ascending=False)
gene_rank = gene_rank.reset_index(drop=True)

res = gseapy.prerank(rnk=gene_rank, gene_sets=gset, 
                    min_size=min_size,
                    max_size=max_size,
                    permutation_num=1000, # reduce number to speed up testing
                    outdir=None, # don't write to disk
                    seed=6,
                    verbose=True,)

res.res2d.sort_values(by=['FDR q-val'], inplace=True, ascending=True)
res.res2d.head(10)

