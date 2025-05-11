import pertpy as pt
import scanpy as sc
import anndata as ad
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import time

# --------------------------
# CONFIGURATION
# --------------------------
input_dir = "/home/ubuntu/frameshift-1/data/filtered_h5ad_allcellines"  # Directory with input .h5ad files
output_dir = "/home/ubuntu/frameshift-1/carlota/augur_allcellines"      # Output folder for results
os.makedirs(output_dir, exist_ok=True)  # Create output folder if it doesn't exist

# --------------------------
# STEP 1: RUN AUGUR ANALYSIS
# --------------------------

# List all .h5ad files in input directory
file_list = [f for f in os.listdir(input_dir) if f.endswith('.h5ad')]

for filename in file_list:
    file_path = os.path.join(input_dir, filename)
    print(f"\n🔵 Processing file: {filename}")

    # Extract plate number from filename (optional, used for file naming)
    match = re.search(r'plate(\d+)', filename)
    plate_number = match.group(1) if match else "unknown"

    # Load data in backed mode (does not fully load into memory yet)
    adata = ad.read_h5ad(file_path, backed='r')

    # Get unique drug treatments, exclude DMSO_TF (used as control)
    drugs = adata.obs['drug'].unique()
    drugs = [drug for drug in drugs if drug != 'DMSO_TF']
    print(f"Drugs found (excluding DMSO_TF): {drugs}")

    # Initialize Augur random forest classifier
    ag_rfc = pt.tl.Augur("random_forest_classifier")

    # Loop over each drug treatment
    for drug_to_use in drugs:
        print(f"🟢 Processing drug: {drug_to_use}")

        # Clean drug name for safe filename usage
        safe_drug_name = re.sub(r'[^A-Za-z0-9\-]+', '_', drug_to_use)
        base_name = f"plate{plate_number}_{safe_drug_name}"

        # Select cells treated with either current drug or control (DMSO_TF)
        selected_cells = adata.obs.index[adata.obs['drug'].isin([drug_to_use, 'DMSO_TF'])]
        adata_subset = adata[selected_cells, :]
        adata_mem = adata_subset.to_memory()  # Load fully into memory for analysis

        # Create 'condition' label (stim vs ctrl)
        adata_mem.obs['condition'] = adata_mem.obs['drug']
        adata_mem.obs["condition"].replace({"DMSO_TF": "ctrl", drug_to_use: "stim"}, inplace=True)

        # Save raw data state for future reference
        adata_mem.raw = adata_mem.copy()

        # Preprocessing: normalize, log-transform, select highly variable genes
        sc.pp.normalize_total(adata_mem, target_sum=1e4)
        sc.pp.log1p(adata_mem)
        sc.pp.highly_variable_genes(adata_mem, flavor="seurat", n_top_genes=2000)

        # Prepare data for Augur
        loaded_data = ag_rfc.load(adata_mem, label_col="condition", cell_type_col="cell_name")

        # Run classification + record runtime
        start_time = time.time()
        v_adata, v_results = ag_rfc.predict(
            loaded_data,
            subsample_size=20,
            n_threads=4,
            select_variance_features=False,
            span=1
        )
        elapsed_time = time.time() - start_time
        print(f"⏳ Prediction took {elapsed_time:.2f} seconds")

        # ----------------------
        # Save outputs per drug
        # ----------------------

        # Save summary metrics table
        df_summary = v_results["summary_metrics"].copy()
        df_summary["drug"] = drug_to_use
        df_summary.to_csv(os.path.join(output_dir, f"summary_metrics_{base_name}.csv"), index=False)

        # Save feature importances table
        df_feat = v_results["feature_importances"].copy()
        df_feat["drug"] = drug_to_use
        df_feat.to_csv(os.path.join(output_dir, f"feature_importances_{base_name}.csv"), index=False)

        # Plot & save lollipop plot
        fig, ax = plt.subplots()
        ag_rfc.plot_lollipop(v_results, ax=ax)
        fig.savefig(os.path.join(output_dir, f"lollipop_{base_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Plot & save important features plot
        fig, ax = plt.subplots()
        ag_rfc.plot_important_features(v_results, ax=ax)
        fig.savefig(os.path.join(output_dir, f"important_features_{base_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Finished drug: {drug_to_use} in plate {plate_number}")

    # Close file after processing
    adata.file.close()

print("\n🎉 All files processed!")

# --------------------------
# STEP 2: COMBINE RESULTS
# --------------------------

print("\n📂 Building combined summary matrix...")

rows = []

# Get all summary metric files
file_list = [file for file in os.listdir(output_dir) if file.startswith("summary_metrics_") and file.endswith(".csv")]

for file in file_list:
    file_path = os.path.join(output_dir, file)

    # Extract plate + drug from filename
    match = re.match(r"summary_metrics_plate(\d+)_(.+)\.csv", file)
    if match:
        plate = match.group(1)
        drug = match.group(2)

        df = pd.read_csv(file_path)
        first_row = df.iloc[0].to_dict()  # Get first row only (Augur summary)

        # Remove redundant 'drug' field
        if "drug" in first_row:
            del first_row["drug"]

        first_row["Plate"] = plate
        first_row["Treatment"] = drug
        rows.append(first_row)

# Build final dataframe
combined_df = pd.DataFrame(rows)

if combined_df.empty:
    print("❌ No data found! Exiting.")
else:
    # Set index for easy sorting
    combined_df = combined_df.set_index(["Plate", "Treatment"]).sort_index()

    # Save final combined summary table
    output_file = os.path.join(output_dir, "augur_combined_summary_matrix.csv")
    combined_df.to_csv(output_file)
    print(f"\n✅ Combined summary matrix saved to: {output_file}")
    print(f"🔢 Shape: {combined_df.shape[0]} plate-drug pairs x {combined_df.shape[1]} scores")

    # --------------------------
    # Optional: create pivoted matrix (treatments x plates)
    # --------------------------
    pivot_df = combined_df.reset_index().pivot(index='Treatment', columns='Plate')
    pivot_df.to_csv(os.path.join(output_dir, "augur_pivot_matrix.csv"))
