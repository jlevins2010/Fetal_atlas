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


# Set parameters

# In[2]:


neighbors = 15
nPCs = 30
leiden_resolution = 1


# In[3]:


adata_CosMx = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/fetal_annotated_AllCells.h5ad")
adata_scRNA = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/MergedFetalOnly_annotated_all_final.h5ad")


# In[4]:


adata_CosMx.obs['Count_percentile'] = adata_CosMx.obs['nCount_RNA'].rank(pct=True) * 100
adata_CosMx.layers["counts"] = adata_CosMx.layers["All_counts"].copy()
adata_CosMx.obs["tech"] = "CosMx"

sc.pl.violin(adata_CosMx, keys = ['UMOD'],  size = 1, groupby = 'sample', rotation= 45, layer = "counts")


# In[5]:


adata_scRNA.obs['Count_percentile'] = adata_scRNA.obs['total_counts'].rank(pct=True) * 100
adata_scRNA.obs["tech"] = "scRNA"

sc.pl.violin(adata_scRNA, keys = ['UMOD'],  size = 1, groupby = 'sample', rotation= 45,use_raw = False, layer = "counts")


# In[6]:


adata_merge = anndata.concat([adata_CosMx,adata_scRNA], join = "outer")

adata_merge


# In[7]:


adata_merge.layers


# In[8]:


sc.pl.violin(adata_merge, keys = ['UMOD'],  size = 1, groupby = 'sample', rotation= 45, layer = "counts")


# In[9]:


adata_merge.X = adata_merge.layers["counts"]

keep_str = ["sample","phase","type", 'cellType','cellType3','tech',"cellType_CosMx_1",'leiden','fov',"cell_ID"]
keep_numer = ["pct_counts_mt","pct_nuc","total_counts","nCount_RNA", "n_genes_by_counts",'CenterX_global_px', 'CenterY_global_px', 'Area','Count_percentile','Mean.DAPI', 'Max.DAPI', 'G2M_score', "S_score"]
keep = keep_str + keep_numer
adata_merge.obs= adata_merge.obs[keep]

# ensure proper data types
for i in keep_str:
    adata_merge.obs[i]= adata_merge.obs[i].astype('str')

for i in keep_numer:
    adata_merge.obs[i]= adata_merge.obs[i].astype('float')
    
ageKey = {'0': '19', '1': '18', '2': '15', 'HK2725': '19', 'HK2716': '15', 'HK2722': '14', 'HK2718': '12', 'HK2723': '20'}
adata_merge.obs["gAge"] = adata_merge.obs['sample'].map(ageKey).astype('category')

sc.pl.violin(adata_merge, keys = ['UMOD'],  size = 1, groupby = 'sample', rotation= 45, layer = "counts")


# In[10]:


adata_merge.raw = adata_merge

adata_merge = adata_merge[:,adata_CosMx.var_names]

systemChecks = adata_merge.var_names.str.startswith('SystemControl')
neg_probes = adata_merge.var_names.str.startswith('Negative')
remove = np.add(neg_probes, systemChecks)
keep = np.invert(remove)

adata_merge = adata_merge[:,keep]


# In[11]:


malat1 = adata_merge.var_names.str.startswith('MALAT1')
mito_genes = adata_merge.var_names.str.startswith('MT-')
hb_genes = adata_merge.var_names.str.contains('^HB[^(P)]')

remove = np.add(mito_genes, malat1)
remove = np.add(remove, hb_genes)

keep = np.invert(remove)

adata_merge = adata_merge[:,keep]


# In[12]:


print(adata_merge)

adata_merge.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/fetal_annotated_AllCells_preSCVI_final.h5ad")


# In[13]:


keep_str = ["sample",'tech','fov',"cell_ID", "gAge"]
keep_numer = ['CenterX_global_px', 'CenterY_global_px', 'Area']
keep = keep_str + keep_numer
adata_merge.obs= adata_merge.obs[keep]

del adata_merge.uns
del adata_merge.obsm


# ensure proper data types
for i in keep_str:
    adata_merge.obs[i]= adata_merge.obs[i].astype('str')

for i in keep_numer:
    adata_merge.obs[i]= adata_merge.obs[i].astype('float')

adata_merge.X = adata_merge.layers["All_counts"]


# In[14]:


layers_to_remove = ['Cytoplasm', 'Membrane', 'Nuclear', 'counts', 'log1p']
for layer_name in layers_to_remove:
    del adata_merge.layers[layer_name]


# In[15]:


adata1 = adata_merge[adata_merge.obs["sample"] == "0"]
adata2 = adata_merge[adata_merge.obs["sample"] == "1"]
adata3 = adata_merge[adata_merge.obs["sample"] == "2"]

samples = [adata1, adata2, adata3]

print(samples)


# In[16]:


adata1.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/HK3524_raw.h5ad")
adata2.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/FK1_raw.h5ad")
adata3.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/FK4_raw.h5ad")


# In[17]:


print(adata1.X)
print(adata1.obs)
print(adata1.var_names)

