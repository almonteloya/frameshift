import pertpy as pt
import scanpy as sc
import anndata as ad
import pandas as pd
import os
import re
import time
import multiprocessing as mp
import traceback

# --------------------------
# CONFIGURATION
# --------------------------
input_dir = "/home/ubuntu/frameshift-1/data/filtered_h5ad_allcellines"
output_dir = "/home/ubuntu/frameshift-1/data/augur_parallel"

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def setup_plate_directories(output_dir, plate_number):
    """Create the directory structure for a plate's output"""
    plate_dir = os.path.join(output_dir, f"plate{plate_number}")
    scores_dir = os.path.join(plate_dir, "scores")
    importance_dir = os.path.join(plate_dir, "importances")
    
    os.makedirs(plate_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(importance_dir, exist_ok=True)
    
    return plate_dir, scores_dir, importance_dir

def process_drug(args):
    """Process a single drug for a given plate"""
    drug_to_use, adata_path, plate_number, output_dir, augur_threads = args
    
    try:
        print(f"🟢 Processing drug: {drug_to_use} for plate {plate_number}")
        
        # Setup directories
        _, scores_dir, importance_dir = setup_plate_directories(output_dir, plate_number)
        
        safe_drug_name = re.sub(r'[^A-Za-z0-9\-]+', '_', drug_to_use)
        base_name = f"{safe_drug_name}"
        
        # Load data
        adata = ad.read_h5ad(adata_path, backed='r')
        selected_cells = adata.obs.index[adata.obs['drug'].isin([drug_to_use, 'DMSO_TF'])]
        adata_subset = adata[selected_cells, :]
        adata_mem = adata_subset.to_memory()
        
        adata_mem.obs['condition'] = adata_mem.obs['drug']
        adata_mem.obs["condition"].replace({"DMSO_TF": "ctrl", drug_to_use: "stim"}, inplace=True)
        
        sc.pp.normalize_total(adata_mem, target_sum=1e4)
        sc.pp.log1p(adata_mem)
        
        # Note: Fixed a bug in the pasted code - should be adata_mem not drug_data
        sc.pp.highly_variable_genes(adata_mem, flavor="seurat", n_top_genes=2000)
        
        # Run Augur
        ag_rfc = pt.tl.Augur("random_forest_classifier")
        loaded_data = ag_rfc.load(adata_mem, label_col="condition", cell_type_col="cell_name")
        
        start_time = time.time()
        
        # Use threading backend to avoid nested multiprocessing issues
        # Also set n_threads to a lower value (4) to enable more parallel drug processing
        v_adata, v_results = ag_rfc.predict(
            loaded_data,
            subsample_size=20,
            n_threads=augur_threads,
            select_variance_features=False,
            span=1,
        )
        elapsed_time = time.time() - start_time
        print(f"⏳ Prediction for {drug_to_use} on plate {plate_number} took {elapsed_time:.2f} seconds")
        
        # Save outputs with organized directory structure
        df_summary = v_results["summary_metrics"]
        df_summary["drug"] = drug_to_use
        df_summary["plate"] = f"plate{plate_number}"
        df_summary.to_csv(os.path.join(scores_dir, f"{base_name}.csv"), index=False)
        
        df_feat = v_results["feature_importances"]
        df_feat["drug"] = drug_to_use
        df_feat["plate"] = f"plate{plate_number}" 
        df_feat.to_csv(os.path.join(importance_dir, f"{base_name}.csv"), index=False)
        
        adata.file.close()
        print(f"✅ Finished drug: {drug_to_use} in plate {plate_number}")
        return True
    except Exception as e:
        print(f"❌ Error processing drug {drug_to_use} for plate {plate_number}: {str(e)}")
        print(traceback.format_exc())
        return False

def process_plates(input_dir, output_dir, plates=None, num_drug_jobs=24, augur_threads=4):
    """Process plates sequentially, but drugs within each plate in parallel"""
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get h5ad files
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.h5ad')]
    
    # Filter plates if specified
    if plates:
        file_list = [f for f in all_files if any(f"plate{p}" in f for p in plates)]
        print(f"Filtering to process {len(file_list)} out of {len(all_files)} plate files")
    else:
        file_list = all_files
        print(f"Found {len(file_list)} plate files to process")
    
    if not file_list:
        print("No files to process. Exiting.")
        return
    
    # Print configuration
    print(f"🖥️ Configuration:")
    print(f"  - Parallel drug jobs per plate: {num_drug_jobs}")
    print(f"  - Augur threads per drug job: {augur_threads}")
    print(f"  - Total CPU cores per plate: approximately {num_drug_jobs * augur_threads}")
    
    print(f"📂 I/O Configuration:")
    print(f"  - Input directory: {input_dir}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Output structure: [output_dir]/plate[X]/[scores|importances]/[drug].csv")
    
    # Process plates one by one
    overall_start_time = time.time()
    processed_plates = 0
    
    # Set multiprocessing start method to 'fork' to better handle threading
    # This helps with the nested parallelism issue
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        # Method might already be set
        pass
    
    for filename in file_list:
        plate_start_time = time.time()
        file_path = os.path.join(input_dir, filename)
        print(f"\n🔵 Processing file: {filename}")
        
        match = re.search(r'plate(\d+)', filename)
        plate_number = match.group(1) if match else "unknown"
        
        # Create directories for this plate
        setup_plate_directories(output_dir, plate_number)
        
        # Get drugs list
        adata = ad.read_h5ad(file_path, backed='r')
        drugs = adata.obs['drug'].unique()
        drugs = [drug for drug in drugs if drug != 'DMSO_TF']
        adata.file.close()
        
        print(f"Drugs found for plate {plate_number} (excluding DMSO_TF): {drugs}")
        
        # Create task arguments for process_drug
        drug_tasks = [(drug, file_path, plate_number, output_dir, augur_threads) for drug in drugs]
        
        # Process drugs in parallel within this plate - increase the number of parallel jobs
        n_jobs = min(num_drug_jobs, len(drugs))
        print(f"Processing {len(drugs)} drugs with {n_jobs} parallel jobs")
        
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(process_drug, drug_tasks)
        
        # Report plate results
        success_count = sum(1 for r in results if r)
        plate_time = time.time() - plate_start_time
        print(f"✅ Completed processing plate {plate_number}: {success_count}/{len(drugs)} drugs successful")
        print(f"⏱️ Plate processing time: {plate_time:.2f} seconds ({plate_time/60:.2f} minutes)")
        
        processed_plates += 1
    
    # Overall summary
    overall_time = time.time() - overall_start_time
    print(f"\n🎉 Processing complete!")
    print(f"  - Successfully processed: {processed_plates} plates")
    print(f"  - Total execution time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")

# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    # Configuration parameters - adjusted for more parallelism with fewer threads per job
    INPUT_DIR = "/home/ubuntu/frameshift-1/data/filtered_h5ad_allcellines"
    OUTPUT_DIR = "/home/ubuntu/frameshift-1/data/augur_parallel"
    NUM_DRUG_JOBS = 48  # Increased number of concurrent drug jobs 
    AUGUR_THREADS = 2  # Reduced threads per job to 4
    
    # Optional: specify specific plates to process (by number)
    # PLATES = [1, 2, 3]  # Process only plates 1, 2, and 3
    PLATES = None  # Process all plates
    
    # Run the analysis
    process_plates(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        plates=PLATES,
        num_drug_jobs=NUM_DRUG_JOBS,
        augur_threads=AUGUR_THREADS
    )