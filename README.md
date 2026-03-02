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

## Name of your model - Methodology

- Summarize in a few sentences the building blocks of your generative GNN model.

- Figure of your model.

## Used External Libraries

- Give instructions on how to install the external libraries if you have used any in your code.

## Results

- Insert your bar plots.


## References

- Do not forget to include the references to methods you used to build your model.
