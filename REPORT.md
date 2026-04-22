# Self-Pruning Neural Network --- Final Report

## 1. Introduction

Modern deep neural networks often contain millions of parameters, many
of which become redundant after training. While overparameterization
improves optimization and generalization, it introduces several
practical challenges:

-   Increased memory consumption
-   Higher inference latency
-   Larger deployment costs
-   Greater energy usage

This project implements a self-pruning neural network that dynamically
removes unnecessary weights during training using learnable gates and L1
sparsity regularization.

Unlike traditional pruning methods that operate after training, this
approach performs online structural adaptation, enabling the model to
automatically learn compact and efficient representations during
optimization.

------------------------------------------------------------------------

## 2. Method Overview

Each weight in the network is associated with a learnable gate:

pruned_weight = weight × sigmoid(gate_score)

If:

sigmoid(gate_score) ≈ 0

the corresponding connection is effectively removed from the model.

This mechanism allows the network to:

-   Retain important connections
-   Suppress redundant weights
-   Learn sparse structures automatically
-   Maintain predictive performance

------------------------------------------------------------------------

## 3. Why L1 Penalty Encourages Sparsity

The total training loss is defined as:

L_total = L_classification + λ × L_sparsity

Where:

L_sparsity = mean(sigmoid(g))

Key mechanism:

-   L1 applies a consistent gradient pushing gate values toward zero
-   Small or unimportant weights are gradually suppressed
-   Important connections remain active due to classification gradients
-   The model converges to a sparse architecture

This interaction produces a bimodal distribution of gate values:

-   A large spike near zero (pruned weights)
-   A smaller cluster of active weights

------------------------------------------------------------------------

## 4. Experimental Setup

Dataset:

-   CIFAR-10 image classification dataset
-   Automatically loaded using torchvision.datasets

Model Architecture:

512 → 256 → 128 → 10

Training Configuration:

-   Optimizer: Adam
-   Learning rate scheduler: Cosine Annealing
-   Precision: Mixed Precision (AMP)
-   Loss function: Cross-Entropy + L1 Sparsity Regularization
-   Sparsity threshold: gate \< 0.01

------------------------------------------------------------------------

## 5. Results

| Lambda | Test Accuracy | Sparsity (%) |
|--------|--------------|-------------|
| 0.5    | 88.45%      | 84.9%       |
| 5      | 87.94%      | 95.3%       |
| 50     | 86.28%      | 99.5%       |

---

## 6. Key Observations

The experiments demonstrate a clear relationship between the sparsity
coefficient (λ) and model behavior.

1.  Sparsity increases with λ.
2.  Accuracy remains stable under moderate sparsity.
3.  Extreme sparsity eventually impacts performance.
4.  Model compression is substantial with minimal accuracy loss.

Accuracy drop from lowest to highest λ:

≈ 2.17%

------------------------------------------------------------------------

## 7. Best Model Selection

Best trade-off:

Lambda = 5

Justification:

-   Very high sparsity (95.3%)
-   Minimal accuracy reduction
-   Stable convergence behavior
-   Efficient model compression

------------------------------------------------------------------------

## 8. Conclusion

The self-pruning neural network successfully demonstrates:

-   Automatic model compression during training
-   High sparsity levels without significant accuracy loss
-   Stable optimization behavior
-   Reduced computational and memory requirements

This approach confirms that dynamic pruning is a practical strategy for
building efficient neural networks suitable for real-world deployment.
