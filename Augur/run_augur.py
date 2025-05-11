import pertpy as pt
import scanpy as sc
import anndata as ad
import os
import matplotlib.pyplot as plt    
import time

import pandas as pd

data_dir = "/home/ubuntu/frameshift-1/data/h5ad"
augur_out = "/home/ubuntu/frameshift-1/data/augur_out"

parquet_path = "/home/ubuntu/frameshift-1/data/metadata/obs_metadata.parquet"
obs_meta = pd.read_parquet(path=parquet_path, engine="pyarrow")

control = "[('DMSO_TF', 0.0, 'uM')]"
for plate in obs_meta["plate"].unique():

	if plate == 'plate10':
		print('Skipping plate10')
		continue

	print(f"Processing plate: {plate}")
	# Load the data
	adata = ad.read_h5ad(f"{data_dir}/{plate}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad", backed="r")
	drugs = adata.obs["drugname_drugconc"].unique()
	drugs = drugs[drugs != control]
	for drug in drugs:
		try:
			ag_rfc = pt.tl.Augur("random_forest_classifier")
			print(drug)
			drug_data = adata[adata.obs["drugname_drugconc"].isin([control, drug])].to_memory()
			drug_data.obs['condition'] = drug_data.obs['drug']
			drug_data.obs["condition"].replace({control: "ctrl", drug: "stim"}, inplace=True)
			sc.pp.normalize_total(drug_data, target_sum=1e6)
			sc.pp.log1p(drug_data)
			sc.pp.highly_variable_genes(drug_data, flavor="seurat", n_top_genes=2000)
			loaded_data = ag_rfc.load(drug_data,label_col="condition", cell_type_col="cell_name")
			v_adata, v_results = ag_rfc.predict(loaded_data, subsample_size=20, n_threads=92, select_variance_features=False, span=1, n_subsamples=20)

			df = v_results["summary_metrics"]
			df["drug"] = drug
			filename = f"{augur_out}/scores/{plate}__{drug}.csv"
			df.to_csv(filename, index=False)
			print("saved!")
			print("saving important features...")
			print(v_results["feature_importances"])
			df = v_results["feature_importances"]
			df["drug"] = drug
			filename = f"{augur_out}/importance/{plate}__{drug}.csv"
			df.to_csv(filename, index=False)
		
		except Exception as e:
			print(f"Error processing drug {drug} on plate {plate}: {e}")