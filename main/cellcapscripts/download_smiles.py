#Import Libraries
from datasets import load_dataset
from scipy.sparse import csr_matrix
import anndata
import pandas as pd
import pubchempy as pcp
import anndata as ad

def create_anndata_from_generator(generator, gene_vocab, sample_size=None):
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items)
    token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

    data, indices, indptr = [], [], [0]
    obs_data = []

    for i, cell in enumerate(generator):
        if sample_size is not None and i >= sample_size:
            break
        genes = cell['genes']
        expressions = cell['expressions']
        if expressions[0] < 0: 
            genes = genes[1:]
            expressions = expressions[1:]

        col_indices = [token_id_to_col_idx[gene] for gene in genes if gene in token_id_to_col_idx]
        valid_expressions = [expr for gene, expr in zip(genes, expressions) if gene in token_id_to_col_idx]

        data.extend(valid_expressions)
        indices.extend(col_indices)
        indptr.append(len(data))

        obs_entry = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
        obs_data.append(obs_entry)

    expr_matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(gene_names)))
    obs_df = pd.DataFrame(obs_data)

    adata = anndata.AnnData(X=expr_matrix, obs=obs_df)
    adata.var.index = pd.Index(gene_names, name='ensembl_id')

    return adata

# tahoe_100m_ds = load_dataset('vevotx/Tahoe-100M', streaming=True, split='train')

# gene_metadata = load_dataset("vevotx/Tahoe-100M", name="gene_metadata", split="train")
# gene_vocab = {entry["token_id"]: entry["ensembl_id"] for entry in gene_metadata}

# adata = create_anndata_from_generator(tahoe_100m_ds, gene_vocab, sample_size=50000)



# adata.obs.head()
# #drug, cell_line_id,drugname_drungconc #
# sample_metadata = load_dataset("vevotx/Tahoe-100M","sample_metadata", split="train").to_pandas()
# adata.obs = pd.merge(adata.obs, sample_metadata.drop(columns=["drug","plate"]), on="sample")
# adata.obs.head()

#We don't need the smiles right now and they are giving the following ERROR: ValueError: Length of passed value for obs_names is 49033, but this AnnData has shape: (50000, 62710)
drug_metadata = load_dataset("vevotx/Tahoe-100M","drug_metadata", split="train").to_pandas()

# adata.obs['drug'] = adata.obs['drug'].astype(str)
# #drugs = [str(c) for c in drugs]

# # Change index
# adata.obs.index = adata.obs.index.astype(str)
# #Subset based off drugs and cell lines
# #adata = adata[adata.obs.drug.isin(drugs)]
# #adata = adata[adata.obs.cell_line_id.isin(cell_lines)]

# #Save adata

# #breakpoint()
# print("writing down file")
drug_metadata.to_csv("../data/drug_metadata.csv")
