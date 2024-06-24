## Fairness-Aware Estimation of Graphical Models

This repository holds the official code for the paper Fairness-Aware Estimation of Graphical Models. 

### 🎯 Abstract
This paper examines the issue of fairness in the estimation of graphical models (GMs), particularly Gaussian, Covariance, and Ising models. These models play a vital role in understanding complex relationships in high-dimensional data. However, standard GMs can result in biased outcomes, especially when the underlying data involves sensitive characteristics or protected groups. To address this, we introduce a comprehensive framework designed to reduce bias in the estimation of GMs related to protected attributes. Our approach involves the integration of the pairwise graph disparity error and a tailored loss function into a nonsmooth multi-objective optimization problem, striving to achieve fairness across different sensitive groups while maintaining the effectiveness of the GMs. Experimental evaluations on synthetic and real-world datasets demonstrate that our framework effectively mitigates bias without undermining GMs' performance.

### 💡 Method
We focus on three types of GMs:
- **Gaussian Graphical Lasso Model (GLasso)**: Rows in the data matrix $\mathbf{X} \in \mathbb{R}^{N \times P}$ are i.i.d. from a multivariate Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$. The conditional independence graph is determined by the sparsity of the inverse covariance matrix $\mathbf{\Sigma}^{-1}\), where \((\mathbf{\Sigma}^{-1})_{jj'} = 0$ indicates conditional independence between the $j$ th and $j’$ th variables.
- **Gaussian Covariance Graph Model (CovGraph)**: Rows are i.i.d. from $\mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$. The marginal independence graph is determined by the sparsity of the covariance matrix $\mathbf{\Sigma}\), where \(\Sigma_{jj'} = 0$ indicates marginal independence between the $j$ th and $j’$ th variables.
- **Binary Ising Graphical Model (BinNet)**: Rows are binary vectors and i.i.d. with
    $p(\mathbf{x}; \boldsymbol{\Theta}) = \left(Z(\boldsymbol{\Theta})\right)^{-1} \exp \big(\sum_{1 \leq j \leq P} \theta_{jj}x_j + \sum_{1 \leq j < j' \leq P} \theta_{jj'}x_jx_{j'} \big).$
    Here, $\boldsymbol{\Theta}$ is a symmetric matrix, and $Z(\boldsymbol{\Theta})$ normalizes the density. $\theta_{jj'} = 0$ indicates conditional independence between the $j$ th and $j’$ th variables. The sparsity pattern of $\boldsymbol{\Theta}$ reflects the conditional independence graph.

### 🗄️ Data
In our paper, we use four real-world datasets listed below.
- **[The Cancer Genome Atlas Program (TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)**: We apply Fair GLasso to analyze RNA-Seq data from TCGA, focusing on lung adenocarcinoma. The data includes expression levels of 60,660 genes from 539 patients. From these, 147 KEGG pathway genes are selected to construct a gene regulatory network. Fair GLasso reveals conditional dependencies, aiding in understanding cancer genetics and identifying therapeutic targets.
- **[Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu)**: ADNI includes standardized uptake value ratios of AV45 and AV1451 tracers in 68 brain regions, as defined by the Desikan-Killiany atlas, collected from 1,143 participants. An amyloid (or tau) accumulation network is constructed to investigate the pattern of amyloid (or tau) accumulation. Fair GLasso are used to uncover conditional dependencies between brain regions, providing insights into Alzheimer's disease progression and identifying potential biomarkers for early diagnosis and treatment response monitoring.
- **[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)**: The performance of Fair CovGraph is evaluated using the Credit Datasets from the UCI Machine Learning Repository. These datasets have been previously used in research on Fair PCA, which shows potential for improvement through sparse covariance estimation.
- **[LFM-1b Dataset](http://www.cp.jku.at/datasets/LFM-1b/)**: LFM-1b Dataset contains over one billion listening events intended for use in recommendation systems. The user-artist play counts dataset is utilized to construct a recommendation network of artists. Our analysis focuses on 80 artists intersecting the 2016 Billboard Artist 100 and 1,807 randomly selected users who listened to at least 400 songs. We transform the play counts into binary datasets for BinNet models, setting play counts above 0 to 1 and all others to 0.

### 📝 Requiremnets
The algorithm is implemented in Python.

### 🔨 Usage
For reproducibility, we provide [synthetic example](https://github.com/PennShenLab/Fair_GMs/blob/main/Simulation%20Study%20Notebook/Experiment_Synthetic_Data_arxive.ipynb) utilizing the [framework](https://github.com/PennShenLab/Fair_GMs/tree/main/GRAPH_Framework-main).

### 🤝 Acknowledgements
The authors Zhuoping Zhou, Davoud Ataee Tarzanagh, and Bojian Hou have contributed equally to this paper.

### 📨 Maintainers
Zhuoping Zhou (zhuopinz@sas.upenn.edu)
