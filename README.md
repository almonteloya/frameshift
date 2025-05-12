# Frameshift Team Submission – Tahoe-DeepDive Hackathon 2025

## Team Name
**Frameshift**

## Members
- Jesus Gonzalez Ferrer, UCSC — [@JesusGF1](https://github.com/JesusGF1)
- Carlota Pereda, UCSF — [@carlotapereda](https://github.com/carlotapereda)
- Laura Almonte, UCSF — [@almonteloya](https://github.com/almonteloya)
- Aidan Winters, Arc Institute/UCSF — [@aidanwinters](https://github.com/aidanwinters)
- Michael Kosicki, LBL — [@lotard](https://github.com/lotard)

---

## Project

### Title
**Defining context-specific responses to drug perturbations in Tahoe 100M dataset**

### Overview
Personalized (i.e. context-specific) treatments lead to better cancer outcomes.  
We want to develop a framework that measures how drugs affect cells differently based on their genetic context, and explains the genetic programs that cells use to respond.  
We define context-specificity as genotype-, cell line-, tissue-of-origin-, and patient-specific effects on gene expression.

### Motivation
Drugs don't work the same way for everyone. Oncotherapies sometimes lack efficacy and tend to be indiscriminate and toxic.  
Broad-acting chemotherapies are effective but are limited by patient side effects.  
We need better ways of stratifying patients, selecting adequate treatments, and simulating adverse effects before they happen.

---

## Methods

### Data Selection
We applied an array of methods to a subset of the Tahoe-100M dataset.  
We focused on cell lines with **KRAS gain-of-function mutations**, especially **G12C**.  
Selected drugs included known KRAS inhibitors, positive controls, and negative controls.

### E-distance
- Used precomputed `scVi` embeddings from Tahoe-100M.
- Calculated distances to plate-paired `DMSO_TF` for each drug and cell line.
- Visualized results.

### MSE
- Applied similar steps as E-distance.
- Started from **pseudobulk samples** provided in the dataset.

### Augur
- A **scRNA classifier** to quantify separability between control and perturbed groups.
- Score of 1 indicates high separability.
- Applied across all cell lines and drug perturbations.

### CellCap
- A **generative model** for perturbation data.
- Models correspondence between basal state and measured perturbation.
- Learns interpretable response programs as weighted gene sets.

---

## Results

- **E-distance** and **MSE** failed to detect context-specific drug effects across selected KRAS cell lines.
- **Augur** and **CellCap**:
  - Detected strong responses in **KRAS-G12C** lines.
  - Captured cell-specific gene expression programs linked to KRAS mutations.

---

## Discussion

The discovery of novel cancer therapies is limited by the lack of generalizable experimental and computational workflows. In a proof-of-concept analysis, we tested four computational methods on the Tahoe-100M dataset for identifying context-specific responses to KRAS inhibitors.

- **Augur** and **CellCap** succeeded in detecting KRAS-inhibitor effects in KRAS-G12C cell lines.
- **E-distance** and **MSE** failed to differentiate responses.

We hypothesize that the success of Augur and CellCap lies in their ability to utilize **local, pathway-level expression** rather than global transcriptomic changes.

Preliminary results highlight genes associated with the **Ras-Raf pathway**, suggesting a targeted effect by the drugs.

### Future Directions
We aim to:
- Scale our approach to all cell lines and drugs in Tahoe-100M.
- Identify potential **cell-type specific drugs**.
- Propose **candidates for clinical development**.


