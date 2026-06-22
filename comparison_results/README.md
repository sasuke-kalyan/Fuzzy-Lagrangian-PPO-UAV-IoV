# Comparison Outputs

Use these outputs for two different purposes:

## 1. Raw Reward Curves

Folder:

```text
comparison_results/graphs/
```

Files:

```text
emergency_comparison.png
suburban_comparison.png
urban_comparison.png
```

Purpose:

```text
Training trend visualization only.
```

These show raw episode reward curves from the saved logs. They should not be used as the main fairness claim because raw reward magnitudes can differ across environment implementations.

## 2. Fair QoS Score Comparison

Folder:

```text
comparison_results/fair_graphs/
```

Files:

```text
emergency_fair_qos_comparison.png
suburban_fair_qos_comparison.png
urban_fair_qos_comparison.png
```

Purpose:

```text
Actual method comparison.
```

These use a shared QoS score based on delay, PDR, signal strength, energy, and the same penalty weights.

## Table

```text
comparison_results/fair_qos_comparison.csv
```

This contains the numeric fair QoS scores used in the fair comparison graphs.
