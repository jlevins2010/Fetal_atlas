#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().system('module list')


# #Previously run: ##do not change!
# Generating an index
# outFile="/home/levinsj/Applications/starIdx" fasta="/home/levinsj/Applications/refdata-cellranger-arc-GRCh38-2020-A-2.0.0/fasta/genome.fa" gtf="/home/levinsj/Applications/refdata-cellranger-arc-GRCh38-2020-A-2.0.0/genes/genes.gtf"
# 
# !STAR --runMode genomeGenerate --genomeDir {outFile}
# --genomeFastaFiles {fasta}
# --sjdbGTFfile {gtf}
# --sjdbOverhang 50 --outFileNamePrefix idx

# In[2]:


#HK2716
#5647-KS-1
genomeDir="/home/levinsj/Applications/starIdx/"
#Read2_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R2_001.fastq.gz"
Read2_fastq_gz="~/Fetal_dir/bustools/5647-KS-1_S1_L001_R2_001.fastq.gz"
#Read1_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R1_001.fastq.gz"
Read1_fastq_gz="~/Fetal_dir/bustools/5647-KS-1_S1_L001_R1_001.fastq.gz"
outputFile="/home/levinsj/Fetal_dir/Velocyto/00_StarSolo/"
sampleName="HK2716"
whiteList="/home/levinsj/Applications/human_GRCh38_gencode.v31.600/10xv3_whitelist.txt"

get_ipython().system('STAR --genomeDir {genomeDir} --readFilesIn {Read2_fastq_gz} {Read1_fastq_gz}  --soloFeatures Gene Velocyto --soloType CB_UMI_Simple  --outFilterScoreMin 30 --soloUMIlen 12 --outFileNamePrefix {outputFile}{sampleName} --soloCBwhitelist {whiteList} --readFilesCommand zcat  --soloBarcodeReadLength 0 --limitOutSJcollapsed 2000000')


# In[3]:


#HK2718
#5566-ZM-2
genomeDir="/home/levinsj/Applications/starIdx/"
#Read2_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R2_001.fastq.gz"
Read2_fastq_gz="~/Fetal_dir/bustools/5566-ZM-2_S01_L005_R2_001.fastq.gz"
#Read1_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R1_001.fastq.gz"
Read1_fastq_gz="~/Fetal_dir/bustools/5566-ZM-2_S01_L005_R1_001.fastq.gz"
outputFile="/home/levinsj/Fetal_dir/Velocyto/00_StarSolo/"
sampleName="HK2718"
whiteList="/home/levinsj/Applications/human_GRCh38_gencode.v31.600/10xv3_whitelist.txt"

get_ipython().system('STAR --genomeDir {genomeDir} --readFilesIn {Read2_fastq_gz} {Read1_fastq_gz}  --soloFeatures Gene Velocyto --soloType CB_UMI_Simple  --outFilterScoreMin 30 --soloUMIlen 12 --outFileNamePrefix {outputFile}{sampleName} --soloCBwhitelist {whiteList} --readFilesCommand zcat  --soloBarcodeReadLength 0 --limitOutSJcollapsed 2000000')


# In[4]:


#HK2722
#5647-KS-2
genomeDir="/home/levinsj/Applications/starIdx/"
#Read2_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R2_001.fastq.gz"
Read2_fastq_gz="~/Fetal_dir/bustools/5647-KS-2_S1_L001_R2_001.fastq.gz"
#Read1_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R1_001.fastq.gz"
Read1_fastq_gz="~/Fetal_dir/bustools/5647-KS-2_S1_L001_R1_001.fastq.gz"
outputFile="/home/levinsj/Fetal_dir/Velocyto/00_StarSolo/"
sampleName="HK2722"
whiteList="/home/levinsj/Applications/human_GRCh38_gencode.v31.600/10xv3_whitelist.txt"

get_ipython().system('STAR --genomeDir {genomeDir} --readFilesIn {Read2_fastq_gz} {Read1_fastq_gz}  --soloFeatures Gene Velocyto --soloType CB_UMI_Simple  --outFilterScoreMin 30 --soloUMIlen 12 --outFileNamePrefix {outputFile}{sampleName} --soloCBwhitelist {whiteList} --readFilesCommand zcat  --soloBarcodeReadLength 0 --limitOutSJcollapsed 2000000')

#--soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts 
#--soloCellFilter
#--soloUMIfiltering MultiGeneUMI_CR
#--soloUMIdedup 1MM_CRR 
#


# In[5]:


#HK2723
#5647-KS-3

genomeDir="/home/levinsj/Applications/starIdx/"
#Read2_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R2_001.fastq.gz"
Read2_fastq_gz="~/Fetal_dir/bustools/5647-KS-3_S1_L001_R2_001.fastq.gz"
#Read1_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R1_001.fastq.gz"
Read1_fastq_gz="~/Fetal_dir/bustools/5647-KS-3_S1_L001_R1_001.fastq.gz"
outputFile="/home/levinsj/Fetal_dir/Velocyto/00_StarSolo/"
sampleName="HK2723"
whiteList="/home/levinsj/Applications/human_GRCh38_gencode.v31.600/10xv3_whitelist.txt"

get_ipython().system('STAR --genomeDir {genomeDir} --readFilesIn {Read2_fastq_gz} {Read1_fastq_gz}  --soloFeatures Gene Velocyto --soloType CB_UMI_Simple  --outFilterScoreMin 30 --soloUMIlen 12 --outFileNamePrefix {outputFile}{sampleName} --soloCBwhitelist {whiteList} --readFilesCommand zcat  --soloBarcodeReadLength 0 --limitOutSJcollapsed 2000000')


# In[6]:


#HK2725
#5738-KS-7

genomeDir="/home/levinsj/Applications/starIdx/"
#Read2_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R2_001.fastq.gz"
Read2_fastq_gz="~/Fetal_dir/bustools/5738-KS-7_S425_L005_R2_001.fastq.gz"
#Read1_fastq_gz="~/Fetal_dir/bustools/EH678AC6P6_HK2770_cell_cntrl_GEX_E8_S6_L001_R1_001.fastq.gz"
Read1_fastq_gz="~/Fetal_dir/bustools/5738-KS-7_S425_L005_R1_001.fastq.gz"
outputFile="/home/levinsj/Fetal_dir/Velocyto/00_StarSolo/"
sampleName="HK2725"
whiteList="/home/levinsj/Applications/human_GRCh38_gencode.v31.600/10xv3_whitelist.txt"

get_ipython().system('STAR --genomeDir {genomeDir} --readFilesIn {Read2_fastq_gz} {Read1_fastq_gz}  --soloFeatures Gene Velocyto --soloType CB_UMI_Simple  --outFilterScoreMin 30 --soloUMIlen 12 --outFileNamePrefix {outputFile}{sampleName} --soloCBwhitelist {whiteList} --readFilesCommand zcat  --soloBarcodeReadLength 0 --limitOutSJcollapsed 2000000')

