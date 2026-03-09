# DGL2026 Brain Graph Super-Resolution Challenge

## Contributors

**Team Name:** Giannis Kitsos  

**Team Members:**
- Giannis Kitsos  
- Kimon Softas  
- Gui Castro  
- Aaliyah Merchant

## Problem Description

Brain graph super-resolution aims to reconstruct a high-resolution (HR) brain connectivity graph (268×268) from its corresponding low-resolution (LR) graph (160×160) for the same subject.

Each graph represents pairwise connectivity between brain regions, encoded as a symmetric weighted adjacency matrix. The challenge is to learn a mapping:

<p align="center">
  f(A<sup>LR</sup>) = Â<sup>HR</sup> ≈ A<sup>HR</sup>
</p>

where:

- A<sup>LR</sup> is the low-resolution adjacency matrix  
- A<sup>HR</sup> is the ground-truth high-resolution matrix  
- Â<sup>HR</sup> is the predicted high-resolution matrix  

### Why is this problem interesting?

- It is a structured prediction task where the output graph has a different resolution (160 nodes → 268 nodes).
- The model must generate an HR connectivity matrix that matches both edge weights and graph-level structure.
- The project requires inductive training, i.e. generalising from training subjects to unseen subjects without transductive information.

The main challenges include:
- Learning cross-resolution mappings (160 → 268 nodes)
- Accurately predicting HR edge weights from LR connectivity patterns
- Maintaining meaningful graph structure under the evaluation measures (e.g. centralities and distributional metrics)

## DEFEND-SR — Methodology

DEFEND-SR extends the DEFEND framework for brain graph super-resolution. It predicts HR edge weights using two parallel low-rank linear branches: one operating directly on the flattened LR edge vector, and another on SGC-precomputed features that capture 2-hop neighbourhood structure. These branches are blended via a learnable sigmoid gate. The initial prediction is then refined by an EfficientLineGraphConv module, which performs message passing in the line graph (dual) space using scatter operations instead of materialising the full ~19M-edge dual graph. The model is trained with a composite loss combining L1 reconstruction and a topology-aware degree-distribution penalty to preserve hub structure.

- Figure of your model.

## Used External Libraries

All required dependencies are listed in the provided `requirements.txt` file.

### Installation

We recommend creating a virtual environment before installing the dependencies.

#### 1. Create a virtual environment

```bash
python -m venv venv
```
#### 2. Activate the virtual environment
On macOS/Linux:
```bash
source venv/bin/activate
```
On Windows:
```bash
venv\Scripts\activate
```
#### 3. Install dependencies
```bash
pip install -r requirements.txt
```
After installation, the environment is ready to run the project.

## Results

<img width="1389" height="985" alt="image" src="https://github.com/user-attachments/assets/0a137cb5-da0f-43b2-9553-1ab59bda1b14" />

## References

- Pala, Singh & Rekik, "Rethinking graph super-resolution: Dual frameworks for topological fidelity," *arXiv preprint arXiv:2511.08853*, 2025.
- Singh & Rekik, "Strongly topology-preserving GNNs for brain graph super-resolution," *Predictive Intelligence in Medicine (PRIME 2024)*, LNCS vol. 15155, Springer, 2025, pp. 124–135.
- Wu, Souza, Zhang, Fifty, Yu & Weinberger, "Simplifying graph convolutional networks," *Proc. 36th Int. Conf. Machine Learning (ICML)*, PMLR, 2019, pp. 6861–6871.
