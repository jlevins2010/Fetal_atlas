#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text


# ### Pseudo Bulk for each cell population against gestational time

# In[2]:


adata = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")


# In[3]:


print(adata)


# In[4]:


cell_identities = {'HK2725': 19.4, 'HK2723': 20.7, 'HK2722': 14.0, 'HK2716': 15.7, 'HK2718': 12.6}
adata.obs["age"] = adata.obs['sample'].map(cell_identities).astype('float')


# ### PT specific Age related gene changes

# In[5]:


from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

PT_subset = adata[adata.obs["cellType"].isin(["PT"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(PT_subset, min_counts=5)

pbs = []
print(PT_subset.obs["age"].value_counts())

for subset in PT_subset.obs["sample"].unique():
    sample_PT = PT_subset[PT_subset.obs["sample"] == subset]
    sample_PT.X = sample_PT.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_PT.X.sum(axis = 0),
                          var = sample_PT.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_PT.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('padj', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/PT_fetal_time_dependent_genes.csv', index=True)


# In[6]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)

ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["HIST1H4C", "GPX3", "RAB7L1", "PTTG1", "APOE", "CUBN"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes PT cells')
plt.show()


# ### LOH genes

# In[7]:


LOH_subset = adata[adata.obs["cellType"].isin(["LOH"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(LOH_subset, min_counts=5)

pbs = []
print(LOH_subset.obs["age"].value_counts())

for subset in LOH_subset.obs["sample"].unique():
    sample_LOH = LOH_subset[LOH_subset.obs["sample"] == subset]
    sample_LOH.X = sample_LOH.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_LOH.X.sum(axis = 0),
                          var = sample_LOH.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_LOH.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('stat', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/LOH_fetal_time_dependent_genes.csv', index=True)


# In[8]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)
ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["SPRY1", "UBL5", "THY1", "SFRP1", "IRX1", "UMOD", "RPS28"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes LOH cells')
plt.show()


# ### DCT

# In[9]:


DCT_subset = adata[adata.obs["cellType"].isin(["DCT"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(DCT_subset, min_counts=5)

pbs = []
print(DCT_subset.obs["age"].value_counts())

for subset in DCT_subset.obs["sample"].unique():
    sample_DCT = DCT_subset[DCT_subset.obs["sample"] == subset]
    sample_DCT.X = sample_DCT.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_DCT.X.sum(axis = 0),
                          var = sample_DCT.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_DCT.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('stat', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/DCT_fetal_time_dependent_genes.csv', index=True)


# In[10]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)
ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["SPRY1", "PAWR", "TMEM258", "IRX1", "RPL10A", "PLCO"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes DCT cells')
plt.show()


# ### NPC

# In[11]:


NPC_subset = adata[adata.obs["cellType"].isin(["NPC"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(NPC_subset, min_counts=5)

pbs = []
print(NPC_subset.obs["age"].value_counts())

for subset in NPC_subset.obs["sample"].unique():
    sample_NPC = NPC_subset[NPC_subset.obs["sample"] == subset]
    sample_NPC.X = sample_NPC.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_NPC.X.sum(axis = 0),
                          var = sample_NPC.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_NPC.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('stat', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/NPC_fetal_time_dependent_genes.csv', index=True)


# In[12]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)
ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["HOXA6", "TMEM258", "IGF2BP1", "SIX2", "UNCX","MEIS2"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes NPC cells')
plt.show()


# ### Podocyte

# In[13]:


Podocyte_subset = adata[adata.obs["cellType"].isin(["Podocyte"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(Podocyte_subset, min_counts=5)

pbs = []
print(Podocyte_subset.obs["age"].value_counts())

for subset in Podocyte_subset.obs["sample"].unique():
    sample_Podocyte = Podocyte_subset[Podocyte_subset.obs["sample"] == subset]
    sample_Podocyte.X = sample_Podocyte.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_Podocyte.X.sum(axis = 0),
                          var = sample_Podocyte.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_Podocyte.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('stat', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/Podocyte_fetal_time_dependent_genes.csv', index=True)


# In[14]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)
ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["SPOCK2", "SLC1A5", "SRP14","PET100", "PTMA", "NPHS2", "WT1", "PTPRO"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes NPC cells')
plt.show()


# ### Stroma

# In[15]:


Stroma_subset = adata[adata.obs["cellType"].isin(["Stroma"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(Stroma_subset, min_counts=5)

pbs = []
print(Stroma_subset.obs["age"].value_counts())

for subset in Stroma_subset.obs["sample"].unique():
    sample_Stroma = Stroma_subset[Stroma_subset.obs["sample"] == subset]
    sample_Stroma.X = sample_Stroma.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_Stroma.X.sum(axis = 0),
                          var = sample_Stroma.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_Stroma.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('stat', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/Stroma_fetal_time_dependent_genes.csv', index=True)


# In[16]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)
ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["FOXD1", "COL6A1", "SRP14","PET100", "HIST1H2AJ", "BPTF", "KDM5A"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes Stroma cells')
plt.show()


# ### Endothelium

# In[17]:


Endothelium_subset = adata[adata.obs["cellType"].isin(["Endothelium"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(Endothelium_subset, min_counts=5)

pbs = []
print(Endothelium_subset.obs["age"].value_counts())

for subset in Endothelium_subset.obs["sample"].unique():
    sample_Endothelium = Endothelium_subset[Endothelium_subset.obs["sample"] == subset]
    sample_Endothelium.X = sample_Endothelium.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_Endothelium.X.sum(axis = 0),
                          var = sample_Endothelium.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_Endothelium.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('stat', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/Endothelium_fetal_time_dependent_genes.csv', index=True)


# In[18]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)
ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["MT-CO1", "NDUFA3", "MT-ND6","PET100", "CEP68", "HSP90AA1", "ADAR", "SLC14A1","EMCN"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes Endothelium cells')
plt.show()


# In[19]:


UB_subset = adata[adata.obs["cellType"].isin(["UB_CT"])]

# Find genes with zero expression in all cells and remove them
sc.pp.filter_genes(UB_subset, min_counts=5)

pbs = []
print(UB_subset.obs["age"].value_counts())

for subset in UB_subset.obs["sample"].unique():
    sample_UB = UB_subset[UB_subset.obs["sample"] == subset]
    sample_UB.X = sample_UB.layers["counts"]
    
    rep_adata = sc.AnnData(X = sample_UB.X.sum(axis = 0),
                          var = sample_UB.var[[]])
    
    rep_adata.obs_names = [subset]
    rep_adata.obs["age"] = sample_UB.obs["age"].iloc[0]

    pbs.append(rep_adata)
                                      
pb = sc.concat(pbs)
int_array = pb.X.astype(int)
counts = pd.DataFrame(int_array, columns = pb.var_names)
pb.age = pb.obs["age"].astype(float)

dds = DeseqDataSet(
    counts=counts,
    metadata=pb.obs,
    design_factors="age",
    continuous_factors=["age"]
)
dds.obs["age"] = dds.obs["age"].astype(float)
dds.deseq2()
stat_res = DeseqStats(dds, contrast =(["age","",""]))
stat_res.summary()
ds_sn = stat_res.results_df
ds_sn = ds_sn.sort_values('stat', ascending = True)
print(ds_sn.head(10))
print(ds_sn.tail(10))

ds_sn.to_csv('/home/levinsj/Fetal_dir/DEG/UB_fetal_time_dependent_genes.csv', index=True)


# In[20]:


ds_sn["nlog10"] = -np.log10(ds_sn["padj"])

plt.figure(figsize=(6,6))
ax = sns.scatterplot(x="log2FoldChange", y="nlog10",  data=ds_sn)
ax.axhline(2, zorder = 0, c = "k", lw = 2, ls = "--")
ax.axvline(0, zorder = 0, c = "k", lw = 2, ls = "--")

plt.legend(loc = 1, bbox_to_anchor = (1.4,1))

texts = []
for i in range(len(ds_sn)):
  if ds_sn.index[i] in ["RET", "AQP2","RAN","SAT1","GATA3","GATA2"]:
    texts.append(plt.text(x= ds_sn.iloc[i].log2FoldChange, y=ds_sn.iloc[i].nlog10, s = ds_sn.index[i]))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.title('Time Associated Genes UB cells')
plt.show()

