# XAI for Cryptocurrency Transaction Analysis using GraphLIME and GNNExplainer

This repository provides implementations of GraphLIME and GNNExplainer models for explainable AI (XAI) on the Elliptic Bitcoin dataset. These models generate explanations for transaction node predictions in terms of important features and relevant subgraphs.


💡 **Objective**

The goal of this project is to apply explainable AI techniques to uncover why certain transactions are classified as illicit or licit. 

The GraphLIME and GNNExplainer models enhance interpretability by identifying key features and visualizing the subgraphs that influence a node's prediction.

This approach increases trust and transparency in graph-based models used for detecting financial crimes such as money laundering.


📁 **Files Included**

The following Google Colab files provide the implementation of GraphLIME and GNNExplainer models for explaining the predictions of cryptocurrency money laundering transactions.


🔹 **GraphLIME.ipynb**

Implements GraphLIME with a GAT (Graph Attention Network) model. It explains important node features for specific transaction nodes.

Steps:
Preprocess the dataset and convert it to PyTorch Geometric format.
Compute node degrees.
Train a GAT model and save accuracy.
Instantiate GraphLIME and explain nodes using explain_node(node_idx, x, edge_index).
Visualize the explanation coefficients to determine feature importance.


🔹 **GNNExplainer.ipynb**

Applies GNNExplainer with both GCN and GAT models. GNNExplainer provides: Single-node explanation as well as Batch explanation for all illicit nodes in a given time step.

Four main sections:
GNNExplainer_GAT: Explains a specific node using GAT.
GNNExplainer_GAT_All: Explains all illicit nodes using GAT.
GNNExplainer_GCN: Explains a specific node using GCN.
GNNExplainer_GCN_All: Explains all illicit nodes using GCN.

Feature importance is visualized using:
explanation.visualize_feature_importance(path, top_k=5)

Subgraph explanation is visualized using:
explanation.visualize_graph(path)


📦 **Dataset:** Elliptic Data Set (Bitcoin Transaction)
Link: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

The Elliptic dataset consists of three components:
Node features: 203,769 Bitcoin transactions, each with 166 features.
Edges: 234,355 directed edges (payment flows between transactions).
Labels: 4,545 labeled as illicit, 42,019 labeled as licit and 157,205 labeled as unknown

Each transaction is a node, and edges represent the flow between them. The data is processed into PyTorch Geometric (PyG) format before model training and explanation.


🛠 **Requirements**

Python 3.7+

PyTorch Geometric

PyTorch

NumPy

Matplotlib


🚀 **How to Run**

Clone this repository.

Open the notebooks:


**Note** that we are considering the dataset to be located at the following path: "/content/drive/MyDrive/elliptic_bitcoin_dataset/"

GraphLIME.ipynb

GNNExplainer.ipynb

Run the cells in sequence to train models and generate explanations.

Visualizations will be saved to the specified output paths.


📬 **Contact**

For any questions or contributions, feel free to open an issue or contact the author.

