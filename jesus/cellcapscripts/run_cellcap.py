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


adata = sc.read_h5ad('../data/tahoe.h5ad')
adata.layers["counts"] = adata.X.copy()

#N drugs used
drugs = list(adata.obs.drug.unique())
n_drugs = len(drugs)


#Covar
covar = list(adata.obs.cell_line_id.unique())
n_covar = len(covar)

#Number of programs used
n_prog = 5


#Drugs are the condition. Condition is string, condition is int
adata.obs["Condition"] = adata.obs.drug.astype(str)
codes, uniques = pd.factorize(adata.obs['Condition'])
uniques = list(uniques)
adata.obs['condition']=codes

adata.obs["Covariate"] = adata.obs.cell_line_id.astype(str)
codescov,uniquescov = pd.factorize(adata.obs['Covariate'])
uniquescov = list(uniquescov)
adata.obs['covariate']=codescov



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
covarY[:,uniquescov.index('CVCL_0480')]=0 #Control cell line
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
cellcap.train(max_epochs=2, batch_size=256)

#get latent representation of basal state
# z = cellcap.get_latent_embedding(adata,batch_size=256)
# adata.obsm['X_basal']=z

z = cellcap.get_latent_embedding(adata,batch_size=256)
_, zH, _ = cellcap.get_pert_embedding(adata)

scaler = MinMaxScaler((0,1))
zH_scaled = scaler.fit_transform(zH)

adata.obsm['X_basal']=z
adata.obsm['X_h_scaled']=zH_scaled


sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_basal', random_state=0, metric='cosine')
sc.tl.draw_graph(adata, layout='fr')

df_usage = plot_program_usage(
    cellcap=cellcap,
    adata=adata,
    perturbation_key='Condition',
)
breakpoint()

attention = ['Q'+str(i) for i in range(1,n_prog+1)]
for d in drugs:
    ad = adata[adata.obs['Condition']==d]
    attn = pd.DataFrame(ad.obsm['X_h_scaled'])
    attn.columns = attention

    for a in attention:
        x = attn[a].values
        ad.obs[a]=x
    
    print(d)
    sc.set_figure_params(scanpy=True, dpi=100, dpi_save=100, vector_friendly=True,figsize=(2,2),fontsize=10)
    sc.pl.draw_graph(ad, color=['Q2','Q4','Q9'],ncols=3, cmap='Oranges',vmin=0,vmax=1,size=30,save=True)

