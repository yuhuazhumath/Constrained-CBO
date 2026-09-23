# Constrained consensus-based optimization

This repository contains MATLAB implementations and numerical experiments for *An interacting particle consensus method for constrained global optimization* by José A. Carrillo, Shi Jin, Haoyu Zhang, and Yuhua Zhu.

## Layout and setup

The code is organized into four directories:

- `algorithms/`: implementations of Algorithms 1 and 2, projected CBO, penalized CBO, and CB2O.
- `examples/`: eight scripts for reproducing the numerical results in the paper.
- `problems/`: functions defining the objectives, constraints, and reference minimizers.
- `utilities/`: helpers.


The experiments use MATLAB R2025b. To set up the repository, open MATLAB, set the current folder to the repository root, and run:

```matlab
setup_repo
```

## Reproducing the paper results

Run the scripts below from the MATLAB editor to reproduce the corresponding figures and tables in the paper.

| Paper result | Script in `examples/` |
| --- | --- |
| Figures 1(b) and 2(a–c) | `preliminary.m` |
| Figures 4–5 and Table 1 | `simple_constraints.m` |
| Figure 6 and Table 2 | `ackley.m` |
| Figure 7 and Table 3 | `thomson.m` |
| Table 4 and Table 7(a,b): sensitivity | `sensitivity.m` |
| Table 5: independent-noise ablation | `noise_ablation.m` |
| Table 6: Thomson conditioning | `conditioning.m` |
| Table 8: runtime comparison | `runtime_comparison.m` |

The scripts save checkpoints, EPS and PNG figures, and CSV tables under `results/`.

Runtime values in Tables 5 and 8 depend on the machine and MATLAB session. Floating-point differences across platforms may also affect the results of stochastic runs.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{carrillo2024interacting,
  title={An interacting particle consensus method for constrained global optimization},
  author={Carrillo, Jos{\'e} A and Jin, Shi and Zhang, Haoyu and Zhu, Yuhua},
  journal={arXiv preprint arXiv:2405.00891},
  year={2024}
}
```
