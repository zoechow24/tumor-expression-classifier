# Cancer Tumor Classification & Dimensionality Reduction

## Project Overview
This project implements a production-grade machine learning pipeline to identify **malignant tumor profiles** from continuous cellular morphometric measurements. In computational oncology and high-throughput genomics, analyzing high-dimensional tissue matrices frequently presents the **"Curse of Dimensionality"**—where features outnumber rows, causing models to overfit on noise. 

This pipeline resolves this engineering constraint. It utilizes **Principal Component Analysis (PCA)** to securely compress a 30-dimensional feature space down to 2 optimized components, then trains a **Linear Support Vector Machine (SVM)** to establish a high-accuracy, clinically sound classification boundary.

The pipeline achieves a **95.61% validation accuracy** while actively optimizing for medical safety.

## Data
The dataset utilized is the classic **Wisconsin Breast Cancer Dataset**, natively integrated into Scikit-Learn (`load_breast_cancer`). It features 569 patient biopsy samples containing 30 continuous floating-point geometric features (e.g., cell radius, smoothness, and area).

## Pipeline
1. **Data Ingestion**: Unpacks the native scikit-learn biological data bundle (`Bunch` object) into a structured Pandas DataFrame. The raw target classifications are mapped immediately to capitalized, human-readable labels (`Malignant` / `Benign`) to enforce upfront code readability and prevent silent target-inversion bugs during downstream analysis.
2. **Exploratory Data Analysis (EDA)**: Conducts a global missing-value audit to verify dataset completeness (0 null rows found) and diagnostics on the clinical class balance (62.74% Benign vs. 37.26% Malignant). Scale profiling across all 30 features unmasks a massive global variance gap (e.g., `mean smoothness` at ~0.10 vs. `mean area` scaling up to 2,501.00), mandating a structural standardization phase.
3. **Leakage-Proof Data Standardization**: Performs an 80:20 stratified train-test split *before* data transformations to prevent data leakage. A `StandardScaler` calculates scaling constants ($\mu=0, \sigma=1$) exclusively on the 455 training rows using `.fit_transform()`. These locked parameters are then applied to the 114 test rows via `.transform()`, perfectly replicating a real-world clinical deployment.
4. **Dimensionality Reduction (PCA)**: Initialized with `n_components=2` to compress the 30 continuous feature columns down into an optimized 2D coordinate space. This mathematical compression retains an impressive **62.64% of the total dataset variance** (PC1: 44.00%, PC2: 18.65%) while stripping away redundant feature noise and exposing a distinct diagonal spatial boundary between tumor classes.
5. **Model Training & Evaluation (SVM)**: Trains a Linear Support Vector Machine to optimize geometric margins. A linear kernel is deliberately selected as the compressed PCA space exhibits clean visual separation, completely eliminating the need for hyper-flexible kernels (like RBF) that risk memorizing noisy border points on limited clinical samples.

## Core Performance Metrics
* **Validation Accuracy**: 95.61%
* **Clinical Model Specificity**: 92.86%
* **Malignant Precision**: 0.95
* **Malignant Recall (Sensitivity)**: 0.93

### Clinical Valuation Analysis
Through a medical oncology lens, minimizing False Negatives (missing an active cancer case) is significantly more vital than avoiding False Positives. Our pipeline optimizes this clinical threshold by achieving a **Malignant Recall of 0.93** (successfully catching 93% of aggressive cancer cases) while maintaining a balanced **Precision of 0.95**. This clinical safety profile is further validated by a **Specificity of 92.86%**, demonstrating that the pipeline is highly adept at identifying benign tissue correctly and preventing unneeded patient anxiety or invasive follow-up procedures.

## 🛠️ Installation & Reproduction
```bash
# Clone the repository
git clone https://github.com/zoechow24/tumor-expression-classifier.git
cd tumor-expression-classifier

# Initialize and activate the isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Run the Jupyter Notebook pipeline
jupyter notebook tumor_classification.ipynb
```
