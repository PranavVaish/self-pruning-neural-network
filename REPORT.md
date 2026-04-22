# Self-Pruning Neural Network --- Final Report

## 1. Introduction

Modern deep neural networks often contain millions of parameters, many
of which are redundant after training. While overparameterization
improves optimization, it increases:

-   Memory usage
-   Inference latency
-   Deployment cost
-   Energy consumption

This project implements a self-pruning neural network that dynamically
removes unnecessary weights during training using learnable gates and L1
sparsity regularization.

Unlike traditional pruning (post-training), this method performs online
structural adaptation, producing compact and efficient models
automatically.

------------------------------------------------------------------------

## 2. Method Overview

Each weight has an associated learnable gate:

pruned_weight = weight × sigmoid(gate_score)

If:

sigmoid(gate_score) ≈ 0

the connection is effectively removed.

------------------------------------------------------------------------

## 3. Why L1 Penalty Encourages Sparsity

Total loss:

L_total = L_classification + λ × L_sparsity

Where:

L_sparsity = mean(sigmoid(g))

Key behavior:

-   L1 applies constant pressure toward zero
-   Redundant weights shrink
-   Important weights remain active
-   Produces sparse architectures

------------------------------------------------------------------------

## 4. Experimental Setup

Dataset:

CIFAR-10

Architecture:

512 → 256 → 128 → 10

Optimizer:

Adam

Scheduler:

Cosine Annealing

Precision:

Mixed Precision (AMP)

Sparsity threshold:

gate \< 0.01

------------------------------------------------------------------------

## 5. Results

  Lambda   Test Accuracy   Sparsity (%)
  -------- --------------- --------------
  0.5      88.45%          84.9%
  5        87.94%          95.3%
  50       86.28%          99.5%

------------------------------------------------------------------------

## 6. Key Observations

As λ increases:

-   Sparsity increases
-   Accuracy decreases slightly
-   Model size reduces significantly

Accuracy drop from lowest to highest λ:

≈ 2.17%

Maximum compression:

≈ 192× reduction in effective model size

------------------------------------------------------------------------

## 7. Best Model Selection

Best trade-off:

Lambda = 5

Because:

-   Very high sparsity
-   Minimal accuracy loss
-   Stable training behavior

------------------------------------------------------------------------

## 8. Conclusion

The self-pruning neural network successfully demonstrates:

-   Dynamic model compression
-   Minimal accuracy degradation
-   Stable optimization
-   Efficient inference

The experiment confirms that learnable gates with L1 regularization
enable automatic sparsity during training.
