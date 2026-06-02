# CM-TransGCN

**CM-TransGCN: A Hybrid Transformer-GCN Framework for circRNA Biomarker Prediction**

## Introduction

CM-TransGCN is a hybrid graph representation learning framework for circRNA biomarker prediction. The model combines the strengths of Transformer-based global dependency modeling and Graph Convolutional Networks (GCNs) for local neighborhood aggregation, enabling the simultaneous capture of long-range contextual interactions and local topological structures within biological networks.

The framework first encodes circRNA sequence semantics and disease similarity information into node representations. A Transformer module is then employed to capture global structural dependencies, followed by a GCN module to refine local neighborhood information. Finally, a multilayer perceptron (MLP) is used to predict potential circRNA-disease associations.

The overall architecture is illustrated in the corresponding manuscript.

---

## Framework

The workflow of CM-TransGCN consists of four stages:

1. Construction of balanced positive and negative samples.
2. Graph representation learning using a Transformer-GCN hybrid network.
3. Pairwise feature construction through embedding concatenation.
4. CircRNA biomarker prediction using an MLP classifier.

---

## Datasets

The experiments were conducted on the following benchmark datasets:

* circAtlas 3.0
* CircR2Disease 2.0
* circRNADisease 2.0

For the circAtlas dataset, a total of:

* 2,576 validated circRNA-disease associations
* 1,968 circRNAs
* 223 diseases

were retained after preprocessing.

---

## Project Structure

```text
CM-TransGCN
│
├── dataset/
│   ├── interaction.csv
│   └── Se_vector64.csv
│
├── generate_balanced_samples.py
├── generate_node_embeddings.py
├── generate_sample_features.py
├── train_mlp_classifier.py
│
├── requirements.txt
└── README.md
```

---

## Input Files

### interaction.csv

Known circRNA-disease associations:

```text
circRNA_ID,disease_ID
```

### Se_vector64.csv

Initial node feature matrix:

```text
Node_ID,Feature_1,...,Feature_64
```

---

## Running the Pipeline

### Step 1: Generate Balanced Samples

Generate negative samples and construct a balanced dataset.

```bash
python generate_balanced_samples.py
```

Output:

```text
NegativeSample.csv
AllCircRNA.csv
Positive_Negative_Samples.csv
```

---

### Step 2: Learn Graph Embeddings

Train the Transformer-GCN encoder and generate node embeddings.

```bash
python generate_node_embeddings.py
```

Output:

```text
node_embeddings.csv
```

---

### Step 3: Construct Pairwise Features

Replace node IDs with embedding vectors and concatenate pairwise representations.

```bash
python generate_sample_features.py
```

Output:

```text
SampleFeatures.csv
```

---

### Step 4: Train the MLP Classifier

Perform five-fold cross-validation and evaluate predictive performance.

```bash
python train_mlp_classifier.py
```

Output:

```text
5_fold_results.txt

ROC_Curve.tif
PR_Curve.tif

Y_pred_fold0.npy
Y_pred_fold1.npy
Y_pred_fold2.npy
Y_pred_fold3.npy
Y_pred_fold4.npy

Y_test_fold0.npy
Y_test_fold1.npy
Y_test_fold2.npy
Y_test_fold3.npy
Y_test_fold4.npy
```

---

## Evaluation Metrics

The following evaluation metrics are reported:

* Accuracy (Acc.)
* Precision (Prec.)
* Recall (Rec.)
* F1-score (F1)
* Matthews Correlation Coefficient (MCC)
* Specificity (Spec.)
* Area Under the ROC Curve (AUC)
* Area Under the Precision–Recall Curve (AUPR)

All results are obtained using five-fold cross-validation.

---

## Model Configuration

### Transformer Module

* TransformerConv Layer
* Multi-head Self-Attention
* Number of Heads: 4

### GCN Module

* GCNConv Layer
* Number of Layers: 1

### Classifier

* Multi-Layer Perceptron (MLP)

### Optimizer

* Adam

---

## Environment

Recommended environment:

```text
Python 3.10
CUDA 11.7
PyTorch 2.0.1
Torch-Geometric 2.5.3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

Random seeds are fixed throughout the experiments to ensure reproducibility.

Five-fold cross-validation is adopted for all benchmark datasets.

---

## License

This project is released under the MIT License.
