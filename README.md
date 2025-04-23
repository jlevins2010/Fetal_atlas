SCRIPT OVERVIEW
To detail how this analysis was constructed, we will detail the function of each script used in the analysis. The overall schematic shows the order of scripts and which were used to create specific figures for our manuscript.

A. Cell Bender 
1.	00_cellBender_adjustMatricies.ipynb
CellBender was used on the raw unfiltered matrix from cellranger output to generate a gene x cell matrix correcting for ambient RNA and identifying empty droplets. 00_cellBender_adjustMatricies.ipynb was used to perform this, and was performed using a GPU for acceleration.
Previously run on google colab. Also compatible with docker://10jll/cellbender38:version2

B. Scanpy Pre-processing
1.	01_PMACSscanpyPipeline_cellBender.ipynb
This script takes in the cellbender corrected matrices and generates an h5ad object. Each sample has doublets identified and removed using scrublet. Then initial QC is performed on each sample. Finally, samples are merged into a single object and integrated using Harmony. DEGs for each leiden cluster are calculated as well. 
Container used: docker://10jll/scanpy:version2
2.	02a_PMACS_annotation_FetalOnly_final.ipynb
This script takes in the harmony integrated object and annotates each cluster. Clusters with poor QC, lacking clear marker genes are removed. 
Container used: docker://10jll/scanpy:version2

C. Cell Rank Initial Analysis for portioning NPC/UB linages
1.	v03a2_scVelo_fetal_nephronAndUB_final.ipynb
Generates RNAvelocity estimates for the shared UB/NPC lineage object, and estimates matrostates.
Container used: docker://10jll/cellrank:version2
2.	v05a2_cellRank_absorbtionProbability_NephronWithUB.ipynb
This script uses the above estimated RNA velocity of just the NPC and UB clusters to calculate the absorption probability for each cell having originated from either the NPC cluster or the UB cluster. This will be used in the next step to partition the NPC lineage from the UB lineage (following two scripts).
Container used: docker://10jll/cellrank:version2


D. Nephron and UB subclustering
1.	02b_PMACS_annotation_FetalOnly_final_Nephron_UB2.ipynb
Uses the estimated probabilities from v05a2_cellRank_absorbtionProbability_NephronWithUB.ipynb and partitions into a UB and NPC cluster. The UB cluster is then subclustered and saved for downstream analysis.
Container used: docker://10jll/cellrank:version2
2.	02b_PMACS_annotation_FetalOnly_final_Nephron_UB.ipynb
Uses the estimated probabilities from v05a2_cellRank_absorbtionProbability_NephronWithUB.ipynb and partitions into a UB and NPC cluster. The NPC cluster is then subclustered and saved for downstream analysis.
Container used: docker://10jll/cellrank:version2

E. CellRank Analysis on NPC lineage
1.	v03b_scVelo_fetal_nephron_final.ipynb
This script uses scVelo to calculate velocity for the merged velocity of just the isolated NPC linage object using a dynamical model.
Container used: docker://10jll/cellrank:version2
2.	v04b_fetal_nephrons_randomWalk.ipynb
Prior to generating a CellRank estimator, we check the relative contributions of each kernel on random walk trajectories using this script. Output from this script is not used for downstream analyses, but rather to show that kernel weighting is not skewing results. 
Container used: docker://10jll/cellrank:version2
3.	v05b_cellRank_absorbtionProbability_nephron.ipynb
We use CellRank to estimate macrostates and calculate absorption probability for each macrostate. Also generate a PAGA-directed graph and GAMs for gene expression across each trajectory.
Container used: docker://10jll/cellrank:version2
4.	v06_selection.ipynb 
We use CellRank to estimate macrostates. We then group macrostates into large categories (renal corpuscle or tubular fate), and calculate absorption probabilities for these grouped macrostates. Lineage driving genes are also calculated. Absorption probabilities are saved in a csv file for further usage and imputation in spatial data.
Container used: docker://10jll/cellrank:version2
5.	v06_self_renew.ipynb
We use CellRank to estimate macrostates. We then group macrostates into large categories (NPC or any differentiated cell), and calculate absorption probabilities for these grouped macrostates. Lineage driving genes are also calculated. Absorption probabilities are saved in a csv file for further usage and imputation in spatial data.
Container used: docker://10jll/cellrank:version2
6.	v06_specification.ipynb
We use CellRank to estimate macrostates, and calculate absorption probabilities for these grouped macrostates. Lineage driving genes are also calculated. Absorption probabilities are saved in a csv file for further usage and imputation in spatial data.
Container used: docker://10jll/cellrank:version2

F. Pseudobulk Differential Gene expression analysis and Pathway analysis
1.	gAgeDEseq.ipynb
Using the cell annotations from 02a_PMACS_annotation_FetalOnly_final.ipynb, we calculate differentially expressed genes using a pseudobulk approach comparing across gestational age.
Container used: docker://10jll/pydeseq:version3
2.	UpsetPlot_fetalTime.ipynb
Using the DEGs calculated by the script above, we examine gene overlap across cell populations.
Container used: docker://10jll/ pymetaneighbor:version3
3.	GSEApy_fetalTime.ipynb
Uses GSEA pathway analysis to examine pathways that are enriched in the DEGs identified in gAgeDEseq.ipynb.
Container used: docker://10jll/pymetaneighbor:version3
4.	CompareDEG_renew.ipynb
Cross compares the DEGs from gAgeDEseq.ipynb with self-renewal lineage drivers from v06_self_renew.ipynb.
Container used: docker://10jll/ pymetaneighbor:version3

G. STAR SOLO gene matrix generation
1.	v00_StarSolo_final.ipynb
To generate a matrix with spliced and un-spliced transcripts for each sample, we used StarSOLO. This script runs STARSOlO for each sample.

H. RNA velocity with scVelo object import
1.	v01_import_scVelo_fetal_final.ipynb
Uses StarSOLO input matrices to generate a h5ad object with STARSOLO matrices for each object. We use the cell barcodes from the scanpy processed anndata to ensure that barcodes match to the annotated objects. Samples are then merged together into a single object.
Container used: docker://10jll/cellrank:version2
2.	v02_scVelo_fetal_final.ipynb
This script uses scVelo to calculate velocity for the merged velocity of all cell population object above using a dynamical model.
Container used: docker://10jll/cellrank:version2

I. CellRank Analysis on UB lineage
1.	v03b_scVelo_fetal_UB_final.ipynb.
This script uses scVelo to calculate velocity for the merged velocity of just the isolated UB linage object using a dynamical model.
Container used: docker://10jll/cellrank:version2
2.	v05b_cellRank_absorbtionProbability_UB.ipynb
We use CellRank to estimate macrostates and calculate absorption probability for each macrostate. Also generate a PAGA-directed graph for the UB lineage.
Container used: docker://10jll/cellrank:version2


J. Preparing Spatial Data
1.	Example_prepareData_flatFiles_images.ipynb
Using exported data from AtoMx for the spatial datasets, this script generates an h5ad object of the entire slide for further usage.
Container used: docker://10jll/spatial:version2
2.	Example_PartitionSamples.ipynb
This script partitions the slide data from the above script into individual samples and generates QC information.
Container used: docker://10jll/spatial:version2
3.	FetalProject_s03_MergeBasicPreprocessing.ipynb
Generates a merged h5ad object of all CosMx samples. Performs basic QC to remove low quality cells and generates a pre-integrated, pre-batch corrected UMAP.
Container used: docker://10jll/spatial:version2
4.	FetalProject_s03b_PrepareSCVI.ipynb
Ensures that all data is in count format (not log-transformed) and merges single cell data from 02a_PMACS_annotation_FetalOnly_final.ipynb
with the CosMx data from FetalProject_s03_MergeBasicPreprocessing.ipynb. Data is then prepared to run via SCVI for further integration.
Container used: docker://10jll/spatial:version2

K. Integration of Spatial Data with scRNAseq and imputation
1.	SCVI_integrate_fetal_2.ipynb
Generates and trains a model to integrate the single cell data with the CosMx data. We then take this model to get a latent representation of the integrated data, which will be used for further clustering and QC.
Performed on google colab. Can also be run with docker://10jll/scvi_cuda12:version5
2.	FetalProject_s04a_postSCVI_preSCANVI_model2.ipynb
Takes the integrated data from above and removes cells with a high intra dataset distance (cells with poor integration). 
Container used: docker://10jll/spatial:version2
3.	SCANVI_integrate_fetal_2_final.ipynb
Performs another SCVI integration on the data after cells with poor integration are filtered out, followed by SCANVI using the original scRNAseq annotations as input labels. Final model is used to calculate latent variables for final integration.
Performed on google colab. Can also be run with docker://10jll/scvi_cuda12:version5
4.	FetalProject_s04b_postSCANVI_post_process_model2.ipynb
Examines the integrated object generated from the model in SCANVI_integrate_fetal_2_final.ipynb.
Container used: docker://10jll/spatial:version2
5.	FetalProject_s04c_postSCANVI_call_spatial_Neighbors_model2.ipynb
Subsets down to individual samples and calculates physical distances between cells to identify physical neighbring cells. Object is then re-merged.
Container used: docker://10jll/spatial:version2
6.	FetalProject_s05_postSCANVI_nephronTranj_model2.ipynb
Using the integrated scRNAseq data, we calculate the estimated absorption probability data for all trajectories in the spatial dataset. Briefly, we use a weighted approach using inter-modality similarity to calculate these values for every cell within the NPC lineage.
Container used: docker://10jll/spatial:version2
7.	FetalProject_s06_post2SCANVI_impute_allGenes_model2.ipynb
Using the integrated scRNAseq data, we calculate imputed gene expression data for all cells in the spatial dataset across all genes within the scRNAseq. 
Container used: docker://10jll/spatial:version2

L. Examination of Histologic Neighborhoods
1.	Overlay_fetal_FK1.ipynb
Using annotations from the aligned histology, cells from the CosMx dataset for FK1 are annotated using the pathologist annotated neighborhoods. This cell:neighborhood dataframe exported as a csv. Run on google colab.
2.	FetalProject_s07_analyze_manual_neighborhoods.ipynb
Examines the pathologist annotations converted above for further plotting and analysis. 
Container used: docker://10jll/spatial:version2
3.	FetalProject_s08_neighborAnalysis_model2.ipynb
Generates GAMs to examine how neighborhood changes along each differentiation trajectory. 
Container used: docker://10jll/cellrank:version2

M. CytoSignal interaction estimation
1.	FetalProject_s09_cytoSingal_Export.ipynb
Converts data so it can be imported into CytoSignal. Done for each sample individually. Note: we use the imputed full transcriptome expression matrix for this.
Container used: docker://10jll/spatial:version2
2.	prepare_FetalCosmx_example_raw.R
Generates a matrix in R with the data exported from above for an example sample.
Container used: docker://10jll/cytosingal:version4
3.	runCytoSignal_sample_imputed.R
Runs CytoSignal for the example sample using the matrix and other input from the scripts above.
Container used: docker://10jll/cytosingal:version4
4.	prepareForPythonImport_all_interactions_all_samples_new.R
Exports the ligand/cell-cell interaction scores for individual cells for later analysis in python.
Container used: docker://10jll/cytosingal:version4
5.	FindCommonInts.R
Identifies LR interactions in common between samples.
Container used: docker://10jll/cytosingal:version4

N. Ligand downstream Analysis
1.	FetalProject_s10_analyze_ligand_score_all.ipynb
Imports the ligand scores from CytoSignal into python. Generates a UMAP using these ligand scores using all CosMx cells. The ligand x cell matrix is then used to subcluster the cells into neighborhoods using leiden clustering and can be used for calculating differentially abundant ligands between neighborhood clusters. This was done for both all ligand/cell-cell interactions and for the subset of those interactions deemed to be “significant” within the dataset by Cytosignal. 
Container used: docker://10jll/spatial:version2
2.	FetalProject_s10_analyze_ligand_score_conserved.ipynb
Same as above, but only uses ligand scores that were part of a significant interaction as determined by CytoSignal.
Container used: docker://10jll/spatial:version2
3.	FetalProject_s11_analyze_ligand_score_conserved_plots2_new.ipynb
Correlates ligand score with trajectory for all NPC lineage cells. Correlations are plotted on a histogram and exported as a csv.
Container used: docker://10jll/spatial:version2
4.	FetalProject_s11_analyze_ligand_score_corr_PT.ipynb
Correlates ligand score with trajectory (PEC or PT) for all PT cells. Correlations are plotted on a histogram and exported as a csv.
Container used: docker://10jll/spatial:version2
5.	FetalProject_s11_analyze_ligand_score_corr_UB.ipynb
Correlates ligand score with trajectory for the UB trajectory. 
Container used: docker://10jll/spatial:version2
6.	Correlation_Histograms.ipynb
Plot histograms on individual ligands/cell-cell interactions with each trajectory.
Run on google colab.
7.	FetalProject_s11_analyze_ligand_score_pyGAM.ipynb
Calculates GAMs examining ligand/cell-cell interaction presence along each trajectory. Container used: docker://10jll/cellrank:version2
8.	FetalProject_s12_neighborAnalysis_ligand.ipynb
Generates GAMs showing how ligand determine neighborhood changes across each trajectory.
Container used: docker://10jll/cellrank:version2
9.	FetalProject_s13_analyze_environment_neighborhoods
Examines expression changes across the ligand derived neighborhoods generating a heatmap for select genes. 
Container used: docker://10jll/spatial:version2

O. Other
1.	Example_FOV_plotting.ipynb
Example wrapper for FOV plotting for expression, or metadata.
Container used: docker://10jll/spatial:version2
2.	Sankey_cosMxPops.ipynb
Samkey plot script for comparison of annotations across CosMx objects (original, and post-SCANVI).
Run using google colab.

