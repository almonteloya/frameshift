import torch

import numpy as np
import pandas as pd
import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import scvi
scvi.settings.seed = 3
import torch

from cellcap.scvi_module import CellCap
from cellcap.validation.plot import plot_adversarial_classifier_roc, plot_program_usage

# drugs = ['Lonafarnib','BI-3406','RMC-6236','Adagrasib',
#                 'Celecoxib','Homoharringtonine','DMSO_TF']
# cell_lines =  ['SW48','SW620','LU65','LS411N']



cpu_device = torch.device('cpu')


adata = sc.read_h5ad('../data/combined_adata.h5ad')
adata.layers["counts"] = adata.X.copy()

print(adata.obs)
#N drugs used
drugs = list(adata.obs.drug.unique())
n_drugs = len(drugs)


#Covar/ #Cell_line_id or cell_name
covar = list(adata.obs.cell_name.unique())
n_covar = len(covar)

#Number of programs used
n_prog = 12

#Drugs are the condition. Condition is string, condition is int
adata.obs["Condition"] = adata.obs.drug.astype(str)
codes, uniques = pd.factorize(adata.obs['Condition'])
uniques = list(uniques)
adata.obs['condition']=codes
#cell_line / cell_name
adata.obs["Covariate"] = adata.obs.cell_name.astype(str)
codescov,uniquescov = pd.factorize(adata.obs['Covariate'])
uniquescov = list(uniquescov)
adata.obs['covariate']=codescov

print(uniquescov)

drug_names = uniques.copy()
#DMSO_TF - Control would be DMSO_TF
#TODO: Not sure of what I am doing here not removing this is right
#drug_names.remove('DMSO_TF')

# one-hot encoding of perturbation information
target_label = np.zeros((len(codes),len(uniques)))
for i in range(len(codes)):
    j = codes[i]
    target_label[i,j]=1
class_weight = target_label.copy()
target_label[:,uniques.index('DMSO_TF')]=0
#DOn't drop zero columns
#target_label = target_label[:,np.sum(target_label,0)>0]
adata.obsm['X_target']=target_label
print(target_label.shape)

#breakpoint()
#create dummpy covariance factor. Here we would have the information on the covariance, same thing that we do for drugs?
#covarY = np.zeros((len(codes),1))
#adata.obsm['X_covar']=covarY
covarY = np.zeros((len(codescov),len(uniquescov)))
for i in range(len(codescov)):
    j = codescov[i]
    covarY[i,j]=1
class_weightcov = covarY.copy()
covarY[:,uniquescov.index('A549')]=0 #Control cell line
#covarY = covarY[:,np.sum(covarY,0)>0]
adata.obsm['X_covar']=covarY
#print(target_label.shape)
print("drugs",n_drugs)
print("covar",n_covar)
print("target_label shape",target_label.shape) #5000 x 94
print("covarY shape", covarY.shape) #50000 x 49

#Todo: Failure between 256 x 94 and 95 x 5 #TODO: 5 might be the problem, with the number of programs
#breakpoint()
CellCap.setup_anndata(adata, layer="counts", target_key='X_target', covar_key='X_covar')


#Lamda for adversarial learning, the bigger the lamda, the bigger the adversarial component. Lamda 1.0 is commonly suggested.
#n_latent is the number of programs, so 12
#cellcap = CellCap(adata, n_latent=5, n_layers=3, n_drug=n_drugs, n_covar=n_covar, ard_kl_weight=1.0, lamda=1)#.to_device("cpu")
cellcap = CellCap(adata, n_drug=n_drugs,n_covar=n_covar,n_prog=n_prog, ard_kl_weight=1.0, lamda=1.0, n_layers=3)

#Max epochs 1000 recommended
cellcap.train(max_epochs=1, batch_size=256)

#get latent representation of basal state
# z = cellcap.get_latent_embedding(adata,batch_size=256)
# adata.obsm['X_basal']=z

z = cellcap.get_latent_embedding(adata,batch_size=256)
_, zH, _ = cellcap.get_pert_embedding(adata)

scaler = MinMaxScaler((0,1))
zH_scaled = scaler.fit_transform(zH)

adata.obsm['X_basal']=z
adata.obsm['X_h_scaled']=zH_scaled
df_usage = plot_program_usage(
    cellcap=cellcap,
    adata=adata,
    perturbation_key='Condition',
)
#breakpoint()
plt.figure(figsize=(14, 6))
sns.heatmap(df_usage, annot=True, cmap="Oranges", fmt=".5f", linewidths=0.5)
plt.title("Condition Score Heatmap")
plt.ylabel("Condition")
plt.xlabel("Feature Index")
plt.tight_layout()
plt.savefig("Gene_module_score_good", dpi=300)
plt.clf()
# attention = ['Q'+str(i) for i in range(1,n_prog+1)]
# for d in drugs:
#     ad = adata[adata.obs['Condition']==d]
#     attn = pd.DataFrame(ad.obsm['X_h_scaled'])
#     attn.columns = attention

#     for a in attention:
#         x = attn[a].values
#         ad.obs[a]=x
    
#     print(d)
#     sc.set_figure_params(scanpy=True, dpi=100, dpi_save=100, vector_friendly=True,figsize=(2,2),fontsize=10)
#     sc.pl.draw_graph(ad, color=['Q2','Q4','Q9'],ncols=3, cmap='Oranges',vmin=0,vmax=1,size=30,save=True)


from scipy.ndimage import gaussian_filter1d

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


#get H
n_prog = cellcap.module.n_prog
h = cellcap.get_h()
h = pd.DataFrame(h)
h.index = drug_names
h.columns = ['Q'+str(i) for i in range(1,(n_prog+1))]
h = h.loc[drug_names,:]

#PCA
pca = PCA(n_components=2)
h_pc = h.values
h_pc = pca.fit_transform(h_pc)


#Distances on PCA space
distances = np.sqrt(h_pc[:,0]**2 + h_pc[:,1]**2)
plt.scatter(h_pc[:,0], h_pc[:,1], c=list(sns.color_palette("Paired"))[:6], 
            marker='.', s=450, edgecolor='white')
plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line through the origin
plt.axvline(0, color='black', linewidth=0.5)  # Vertical line through the origin

# Connect points to the origin
num_points = h_pc.shape[0]
for i in range(num_points):
    plt.plot([0, h_pc[:,0][i]], [0, h_pc[:,1][i]], 'grey', linewidth=0.75,alpha=0.25)
    
label_offset = 0.05  # Offset for label position
for i in range(num_points):
    plt.text(h_pc[:,0][i]- label_offset*2, h_pc[:,1][i]+ label_offset, h.index[i], fontsize=10)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig("PCA_representation.png",dpi=300)
plt.clf()


print("Starting UMAP plots")
import umap
df = pd.DataFrame(adata.obsm["X_basal"], index=adata.obs_names)
reducer = umap.UMAP()
umap_target = reducer.fit_transform(df)
umap_target = pd.DataFrame(umap_target, columns=["UMAP1", "UMAP2"], index=df.index)
# Add 'drug' and 'cell_line_id' columns from adata.obs
#breakpoint()
umap_target["drug"] = adata.obs["drug"].values
umap_target["cell_name"] = adata.obs["cell_name"].values

#Plot 
drugs = umap_target["drug"].unique()
colors = plt.cm.tab20.colors  # Up to 20 distinct colors
color_map = {drug: colors[i % len(colors)] for i, drug in enumerate(drugs)}

plt.figure(figsize=(10, 7))
for drug in drugs:
    subset = umap_target[umap_target["drug"] == drug]
    plt.scatter(subset["UMAP1"], subset["UMAP2"], label=drug, alpha=0.7, s=20, color=color_map[drug])

plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title="Drug")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP of Drug Perturbation Embedding")
plt.savefig("UMAP_representationdrug.png")
plt.savefig("UMAP_representationcelline.png")

#Identify top perturbed genes:
from cellcap.utils import cosine_distance, identify_top_perturbed_genes

#We use control group to scale whole data
control = adata[adata.obs['timepoint']=='UT']
X = np.asarray(control.X.todense())
scaler = StandardScaler()
X = scaler.fit(X)
adata.layers['scaled'] = scaler.transform(np.asarray(adata.X.todense()))

#get gene loadings
prog_embedding = cellcap.get_resp_loadings()
weights = cellcap.get_loadings()

prog_loading = np.matmul(weights,prog_embedding.T)
prog_loading = pd.DataFrame(prog_loading)
prog_loading.index = adata.var.index

#program of interest based on perturbation
prog_index = 6

#get significant perturbed genes in program 6
w = identify_top_perturbed_genes(pert_loading=prog_loading,prog_index=prog_index)

Pgene = w[np.logical_and(w['Zscore']>0, w['Pval']<0.05)]
Pgene = Pgene.sort_values(by=['Pval'],ascending=True)
Pgene = Pgene.index.tolist()


pert = adata[adata.obs['timepoint']=='3hCA']
X = np.asarray(pert.layers['scaled'])
X = pd.DataFrame(X)
X.columns = pert.var.index
X = X.loc[:,Pgene]
y = pert.obsm['X_h_scaled'][:,(prog_index-1)]

k = 10
topk = SelectKBest(f_regression, k=k).fit(X, y)

top_feature_indices = topk.get_support(indices=True)
newX = X.iloc[:,top_feature_indices]

reg = linear_model.BayesianRidge(fit_intercept=False)
reg.fit(newX, y)

gene_weights = pd.DataFrame(reg.coef_)
gene_weights.index = newX.columns
selected_genes = gene_weights[gene_weights[0]>0][0].sort_values(ascending = False).index.tolist()
import torch

import numpy as np
import pandas as pd
import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import scvi
scvi.settings.seed = 3
import torch

from cellcap.scvi_module import CellCap
from cellcap.validation.plot import plot_adversarial_classifier_roc, plot_program_usage

# drugs = ['Lonafarnib','BI-3406','RMC-6236','Adagrasib',
#                 'Celecoxib','Homoharringtonine','DMSO_TF']
# cell_lines =  ['SW48','SW620','LU65','LS411N']



cpu_device = torch.device('cpu')


adata = sc.read_h5ad('../data/combined_adata.h5ad')
adata.layers["counts"] = adata.X.copy()

print(adata.obs)
#N drugs used
drugs = list(adata.obs.drug.unique())
n_drugs = len(drugs)


#Covar/ #Cell_line_id or cell_name
covar = list(adata.obs.cell_name.unique())
n_covar = len(covar)

#Number of programs used
n_prog = 12


#Drugs are the condition. Condition is string, condition is int
adata.obs["Condition"] = adata.obs.drug.astype(str)
codes, uniques = pd.factorize(adata.obs['Condition'])
uniques = list(uniques)
adata.obs['condition']=codes
#cell_line / cell_name
adata.obs["Covariate"] = adata.obs.cell_name.astype(str)
codescov,uniquescov = pd.factorize(adata.obs['Covariate'])
uniquescov = list(uniquescov)
adata.obs['covariate']=codescov

print(uniquescov)

drug_names = uniques.copy()
#DMSO_TF - Control would be DMSO_TF
#TODO: Not sure of what I am doing here not removing this is right
#drug_names.remove('DMSO_TF')

# one-hot encoding of perturbation information
target_label = np.zeros((len(codes),len(uniques)))
for i in range(len(codes)):
    j = codes[i]
    target_label[i,j]=1
class_weight = target_label.copy()
target_label[:,uniques.index('DMSO_TF')]=0
#DOn't drop zero columns
#target_label = target_label[:,np.sum(target_label,0)>0]
adata.obsm['X_target']=target_label
print(target_label.shape)

#breakpoint()
#create dummpy covariance factor. Here we would have the information on the covariance, same thing that we do for drugs?
#covarY = np.zeros((len(codes),1))
#adata.obsm['X_covar']=covarY
covarY = np.zeros((len(codescov),len(uniquescov)))
for i in range(len(codescov)):
    j = codescov[i]
    covarY[i,j]=1
class_weightcov = covarY.copy()
covarY[:,uniquescov.index('A549')]=0 #Control cell line
#covarY = covarY[:,np.sum(covarY,0)>0]
adata.obsm['X_covar']=covarY
#print(target_label.shape)
print("drugs",n_drugs)
print("covar",n_covar)
print("target_label shape",target_label.shape) #5000 x 94
print("covarY shape", covarY.shape) #50000 x 49

#Todo: Failure between 256 x 94 and 95 x 5 #TODO: 5 might be the problem, with the number of programs
#breakpoint()
CellCap.setup_anndata(adata, layer="counts", target_key='X_target', covar_key='X_covar')


#Lamda for adversarial learning, the bigger the lamda, the bigger the adversarial component. Lamda 1.0 is commonly suggested.
#n_latent is the number of programs, so 12
#cellcap = CellCap(adata, n_latent=5, n_layers=3, n_drug=n_drugs, n_covar=n_covar, ard_kl_weight=1.0, lamda=1)#.to_device("cpu")
cellcap = CellCap(adata, n_drug=n_drugs,n_covar=n_covar,n_prog=n_prog, ard_kl_weight=1.0, lamda=1.0, n_layers=3)

#Max epochs 1000 recommended
cellcap.train(max_epochs=150, batch_size=256)

#get latent representation of basal state
# z = cellcap.get_latent_embedding(adata,batch_size=256)
# adata.obsm['X_basal']=z

z = cellcap.get_latent_embedding(adata,batch_size=256)
_, zH, _ = cellcap.get_pert_embedding(adata)

scaler = MinMaxScaler((0,1))
zH_scaled = scaler.fit_transform(zH)

adata.obsm['X_basal']=z
adata.obsm['X_h_scaled']=zH_scaled
df_usage = plot_program_usage(
    cellcap=cellcap,
    adata=adata,
    perturbation_key='Condition',
)
#breakpoint()
plt.figure(figsize=(14, 6))
sns.heatmap(df_usage, annot=True, cmap="Oranges", fmt=".5f", linewidths=0.5)
plt.title("Condition Score Heatmap")
plt.ylabel("Condition")
plt.xlabel("Feature Index")
plt.tight_layout()
plt.savefig("Gene_module_score_good", dpi=300)
plt.clf()
# attention = ['Q'+str(i) for i in range(1,n_prog+1)]
# for d in drugs:
#     ad = adata[adata.obs['Condition']==d]
#     attn = pd.DataFrame(ad.obsm['X_h_scaled'])
#     attn.columns = attention

#     for a in attention:
#         x = attn[a].values
#         ad.obs[a]=x
    
#     print(d)
#     sc.set_figure_params(scanpy=True, dpi=100, dpi_save=100, vector_friendly=True,figsize=(2,2),fontsize=10)
#     sc.pl.draw_graph(ad, color=['Q2','Q4','Q9'],ncols=3, cmap='Oranges',vmin=0,vmax=1,size=30,save=True)


from scipy.ndimage import gaussian_filter1d

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


#get H
n_prog = cellcap.module.n_prog
h = cellcap.get_h()
h = pd.DataFrame(h)
h.index = drug_names
h.columns = ['Q'+str(i) for i in range(1,(n_prog+1))]
h = h.loc[drug_names,:]

#PCA
pca = PCA(n_components=2)
h_pc = h.values
h_pc = pca.fit_transform(h_pc)


#Distances on PCA space
distances = np.sqrt(h_pc[:,0]**2 + h_pc[:,1]**2)
plt.scatter(h_pc[:,0], h_pc[:,1], c=list(sns.color_palette("Paired"))[:6], 
            marker='.', s=450, edgecolor='white')
plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line through the origin
plt.axvline(0, color='black', linewidth=0.5)  # Vertical line through the origin

# Connect points to the origin
num_points = h_pc.shape[0]
for i in range(num_points):
    plt.plot([0, h_pc[:,0][i]], [0, h_pc[:,1][i]], 'grey', linewidth=0.75,alpha=0.25)
    
label_offset = 0.05  # Offset for label position
for i in range(num_points):
    plt.text(h_pc[:,0][i]- label_offset*2, h_pc[:,1][i]+ label_offset, h.index[i], fontsize=10)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig("PCA_representation.png",dpi=300)
plt.clf()


print("Starting UMAP plots")
import umap
df = pd.DataFrame(adata.obsm["X_basal"], index=adata.obs_names)
reducer = umap.UMAP()
umap_target = reducer.fit_transform(df)
umap_target = pd.DataFrame(umap_target, columns=["UMAP1", "UMAP2"], index=df.index)
# Add 'drug' and 'cell_line_id' columns from adata.obs
#breakpoint()
umap_target["drug"] = adata.obs["drug"].values
umap_target["cell_name"] = adata.obs["cell_name"].values

#Plot 
drugs = umap_target["drug"].unique()
colors = plt.cm.tab20.colors  # Up to 20 distinct colors
color_map = {drug: colors[i % len(colors)] for i, drug in enumerate(drugs)}

plt.figure(figsize=(10, 7))
for drug in drugs:
    subset = umap_target[umap_target["drug"] == drug]
    plt.scatter(subset["UMAP1"], subset["UMAP2"], label=drug, alpha=0.7, s=20, color=color_map[drug])

plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title="Drug")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP of Drug Perturbation Embedding")
plt.savefig("UMAP_representationdrug.png")
# Add 'cell_name' to UMAP DataFrame
umap_target["cell_name"] = adata.obs["cell_name"].values
plt.clf()
# Unique cell names
cell_names = umap_target["cell_name"].unique()

# Assign distinct colors
colors = plt.cm.tab20.colors  # 20-color palette
color_map = {cell: colors[i % len(colors)] for i, cell in enumerate(cell_names)}

# Plot
plt.figure(figsize=(10, 7))
for cell in cell_names:
    subset = umap_target[umap_target["cell_name"] == cell]
    plt.scatter(subset["UMAP1"], subset["UMAP2"], label=cell, alpha=0.7, s=20, color=color_map[cell])

plt.legend(loc="best", bbox_to_anchor=(1.05, 1), title="Cell Name")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP of Cell Name Embedding")
plt.tight_layout()
plt.savefig("UMAP_representationcelline.png")
plt.clf()
#Save umap as csv
umap_target.to_csv("UMAP_Coordinates.csv")

#Identify top perturbed genes:
from cellcap.utils import cosine_distance, identify_top_perturbed_genes

#We use control group to scale whole data
control = adata[adata.obs['drug']=='DMSO_TF']
X = np.asarray(control.X.todense())
scaler = StandardScaler()
X = scaler.fit(X)
adata.layers['scaled'] = scaler.transform(np.asarray(adata.X.todense()))

#get gene loadings
prog_embedding = cellcap.get_resp_loadings()
#Program embeddings
#prog_embedding.to_csv("program_embeddings.csv")
weights = cellcap.get_loadings()
#weighs.to_csv("loadings.csv")
prog_loading = np.matmul(weights,prog_embedding.T)
prog_loading = pd.DataFrame(prog_loading)
prog_loading.index = adata.var.index
prog_loading.to_csv("Program_loadings.csv")

print("Starting program of interest analysis")
import os

# Ensure output directory exists
os.makedirs("program_outputs", exist_ok=True)


all_results = {}

for prog_index in range(1, n_prog + 1):  # Assuming 1-indexed programs
    print(f"Processing program {prog_index}")

    # 1. Identify significantly perturbed genes
    w = identify_top_perturbed_genes(pert_loading=prog_loading, prog_index=prog_index)
    Pgene = w[np.logical_and(w['Zscore'] > 0, w['Pval'] < 0.05)]
    Pgene = Pgene.sort_values(by=['Pval'], ascending=True).index.tolist()

    # 2. Prepare data for regression
    pert = adata.copy()
    X = pd.DataFrame(np.asarray(pert.layers['scaled']), columns=pert.var.index)
    X = X.loc[:, Pgene]
    y = pert.obsm['X_h_scaled'][:, prog_index - 1]

    # 3. Feature selection
    topk = SelectKBest(f_regression, k=10).fit(X, y)
    top_feature_indices = topk.get_support(indices=True)
    newX = X.iloc[:, top_feature_indices]

    # 4. Fit Bayesian Ridge model
    reg = linear_model.BayesianRidge(fit_intercept=False)
    reg.fit(newX, y)
    gene_weights = pd.DataFrame(reg.coef_, index=newX.columns, columns=["Weight"])
    selected_genes = gene_weights[gene_weights["Weight"] > 0].sort_values(by="Weight", ascending=False).index.tolist()

    # 5. Save results
    pert.obs[f"Response_prog{prog_index}"] = y
    umap_target[f"Response_prog{prog_index}"] = y
    all_results[prog_index] = {
        "selected_genes": selected_genes,
        "gene_weights": gene_weights,
        "response": y,
    }

    # 6. Plot UMAP
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        umap_target["UMAP1"],
        umap_target["UMAP2"],
        c=umap_target[f"Response_prog{prog_index}"],
        cmap="viridis",
        s=20,
        alpha=0.8
    )
    plt.colorbar(scatter, label="Response")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"UMAP Colored by Response (Program {prog_index})")
    plt.tight_layout()
    plt.savefig(f"program_outputs/UMAP_response_prog{prog_index}.png", dpi=300)
    plt.close()
    plt.clf()

    expr = X.loc[:,selected_genes]
    expr['Response']=y
    expr = expr.sort_values(by=['Response'],ascending=False)

    x = np.asarray(expr.values[:,:-1]).squeeze()
    x = gaussian_filter1d(x, 1, axis=0, mode='nearest')
    sc.set_figure_params(scanpy=True, dpi=100, dpi_save=100, vector_friendly=True, figsize=(3,x.shape[1]*0.2))
    sns.set_theme(style='white', font_scale=0.75)
    im = plt.imshow(x.T, cmap="PiYG", vmin=-2, vmax=2,aspect='auto',interpolation='nearest')
    plt.grid(False)
    plt.xticks(
        []
    )
    plt.yticks(
        ticks=range(len(expr.columns)-1),
        labels=expr.columns.tolist()[:-1],
    )
    plt.colorbar(im, fraction=0.05, pad=0.04)
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"program_outputs/heatmap_expression{prog_index}.png", dpi=300, bbox_inches="tight")
    plt.clf()




#program of interest based on perturbation
# prog_index = 6
# breakpoint()
# #get significant perturbed genes in program 6
# w = identify_top_perturbed_genes(pert_loading=prog_loading,prog_index=prog_index)

# Pgene = w[np.logical_and(w['Zscore']>0, w['Pval']<0.05)]
# Pgene = Pgene.sort_values(by=['Pval'],ascending=True)
# Pgene = Pgene.index.tolist()


# pert = adata.copy()#[adata.obs['drug']=='DMSO_TF']
# X = np.asarray(pert.layers['scaled'])
# X = pd.DataFrame(X)
# X.columns = pert.var.index
# X = X.loc[:,Pgene]
# y = pert.obsm['X_h_scaled'][:,(prog_index-1)]

# k = 10
# topk = SelectKBest(f_regression, k=k).fit(X, y)

# top_feature_indices = topk.get_support(indices=True)
# newX = X.iloc[:,top_feature_indices]

# reg = linear_model.BayesianRidge(fit_intercept=False)
# reg.fit(newX, y)

# gene_weights = pd.DataFrame(reg.coef_)
# gene_weights.index = newX.columns
# selected_genes = gene_weights[gene_weights[0]>0][0].sort_values(ascending = False).index.tolist()


# pert.obs['Response']=np.asarray(y)

# #Add this to drug and other plot in UMAP. This will most likely fail because we are losing the significance
# umap_target["Response"] = pert.obs['Response'].values

# plt.figure(figsize=(10, 7))
# scatter = plt.scatter(
#     umap_target["UMAP1"],
#     umap_target["UMAP2"],
#     c=umap_target["Response"],
#     cmap="viridis",       # or 'plasma', 'coolwarm', 'Oranges', etc.
#     s=20,
#     alpha=0.8
# )

# plt.colorbar(scatter, label="Response")
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.title("UMAP Colored by Response Value")
# plt.tight_layout()
# plt.savefig("UMAP_response_gradient.png", dpi=300)