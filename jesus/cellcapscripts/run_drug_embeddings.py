import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import umap

sns.set(style='white', font_scale=0.9)

# Load ChemBERTa and tokenizer
chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

# Load dataset with columns: 'drug', 'canonical_smiles', 'target'
data = pd.read_csv("../data/drug_metadata.csv")
#smiles_list = data["canonical_smiles"].tolist()
smiles_list = data["canonical_smiles"].dropna().astype(str).tolist()

# Encode SMILES strings with padding
with torch.no_grad():
    encoded = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True)
    output = chemberta(**encoded)
    # Use [CLS] token representation
    embeddings_cls = output[0][:, 0, :]

# UMAP reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
umap_emb = reducer.fit_transform(embeddings_cls.numpy())

# Create a DataFrame for plotting
umap_df = pd.DataFrame(umap_emb, columns=["UMAP1", "UMAP2"])
umap_df["drug"] = data["drug"].reindex(umap_df.index).values
#umap_df["targets"] = data["targets"].reindex(umap_df.index).values
#breakpoint()
# umap_df["drug"] = data["drug"].values
# umap_df["target"] = data["target"].values
drugs = ['RMC-6236', 'Adagrasib', 'Celecoxib', 'Homoharringtonine', 'Dinaciclib', 'DMSO_TF']

# Set up figure
plt.figure(figsize=(10, 7))

# Plot all cells in light blue
plt.scatter(
    umap_df["UMAP1"],
    umap_df["UMAP2"],
    color="blue",
    alpha=0.2,
    s=20,
    label="Other drugs"
)

# Plot selected drugs in red
highlight = umap_df[umap_df["drug"].isin(drugs)]
plt.scatter(
    highlight["UMAP1"],
    highlight["UMAP2"],
    color="red",
    alpha=0.8,
    s=20,
    label="Selected drugs"
)

# Add labels for each red-highlighted cell
for idx, row in highlight.iterrows():
    plt.text(
        row["UMAP1"], 
        row["UMAP2"] + 0.2,   # slightly above the point
        row["drug"], 
        fontsize=8, 
        color='black', 
        ha='center'
    )

# Format the plot
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP of ChemBERTa Embeddings with Highlighted Drugs")
plt.legend()
plt.tight_layout()
plt.savefig("umap_highlight_selected_drugs.png", dpi=300)
plt.show()

# Plot colored by target
# plt.figure(figsize=(10, 7))
# sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue="targets", s=40, alpha=0.8)
# plt.title("UMAP of ChemBERTa Embeddings Colored by Target")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Target")
# plt.tight_layout()
# plt.savefig("umap_by_target_smiles.png", dpi=300)
# plt.show()