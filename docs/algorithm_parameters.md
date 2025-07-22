# Algorithm Hyperparameters Documentation

This document provides comprehensive documentation of all tunable hyperparameters in the Qloo Supermarket Layout Optimizer algorithms. Understanding and properly tuning these parameters is crucial for optimal performance.

## Table of Contents

1. [Association Rule Mining Parameters](#association-rule-mining-parameters)
2. [Layout Optimization Parameters](#layout-optimization-parameters)
3. [Heuristic Optimization Parameters](#heuristic-optimization-parameters)
4. [Scoring Algorithm Parameters](#scoring-algorithm-parameters)
5. [Performance Parameters](#performance-parameters)
6. [Parameter Tuning Guidelines](#parameter-tuning-guidelines)
7. [Configuration Examples](#configuration-examples)

## Association Rule Mining Parameters

### Apriori Algorithm Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_support` | float | 0.03 | 0.001-0.5 | Minimum support threshold for frequent itemsets. Lower values find more rules but increase computation time. |
| `min_confidence` | float | 0.6 | 0.1-0.95 | Minimum confidence threshold for association rules. Higher values produce more reliable rules. |
| `min_lift` | float | 1.1 | 1.0-5.0 | Minimum lift value for meaningful associations. Values > 1.0 indicate positive correlation. |
| `max_itemset_size` | int | 4 | 2-8 | Maximum size of itemsets to consider. Larger values increase computation exponentially. |

#### Tuning Guidelines:

- **High-volume stores**: Use higher `min_support` (0.05-0.1) to focus on strongest patterns
- **Diverse product mix**: Lower `min_support` (0.01-0.03) to capture niche associations
- **Quality-focused**: Higher `min_confidence` (0.7-0.9) for reliable recommendations
- **Discovery-focused**: Lower `min_confidence` (0.4-0.6) to find emerging patterns

```python
# Example configuration for high-volume store
association_config = {
    'min_support': 0.05,
    'min_confidence': 0.75,
    'min_lift': 1.3,
    'max_itemset_size': 3
}
```

## Layout Optimization Parameters

### Section-Level Optimization

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `target_utilization` | float | 0.85 | 0.6-0.95 | Target utilization rate for store sections. Balances efficiency with accessibility. |
| `min_performance_threshold` | float | 0.7 | 0.5-0.9 | Minimum performance score to trigger optimization. Lower values = more aggressive optimization. |
| `rebalance_frequency_days` | int | 7 | 1-30 | How often to rebalance sections (in days). More frequent = more responsive to changes. |

### Multi-Objective Weights

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `revenue_optimization` | float | 0.4 | 0.1-0.8 | Weight for revenue-focused objectives in multi-objective optimization. |
| `customer_experience` | float | 0.3 | 0.1-0.6 | Weight for customer flow and convenience factors. |
| `operational_efficiency` | float | 0.2 | 0.1-0.5 | Weight for stocking and maintenance ease. |
| `space_utilization` | float | 0.1 | 0.05-0.3 | Weight for efficient space usage. |

*Note: Weights should sum to 1.0*

```yaml
# Example: Customer-focused configuration
objective_weights:
  revenue_optimization: 0.3
  customer_experience: 0.5
  operational_efficiency: 0.15
  space_utilization: 0.05
```

## Heuristic Optimization Parameters

### Crowding Penalty Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `base_penalty` | float | 0.15 | 0.05-0.5 | Base penalty applied to overcrowded sections. |
| `utilization_threshold` | float | 0.9 | 0.7-0.98 | Utilization level where penalty starts applying. |
| `max_penalty` | float | 0.5 | 0.2-1.0 | Maximum penalty multiplier for severe crowding. |
| `exponential_scaling` | float | 2.0 | 1.0-4.0 | Exponential factor for penalty scaling. Higher = steeper penalty curve. |

### Adjacency Reward Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `base_reward` | float | 0.1 | 0.05-0.3 | Base reward for good product adjacency. |
| `association_threshold` | float | 1.5 | 1.1-3.0 | Minimum lift required for adjacency consideration. |
| `max_reward` | float | 0.3 | 0.1-0.8 | Maximum reward multiplier for optimal adjacency. |
| `distance_decay` | float | 0.8 | 0.5-0.95 | How quickly reward decays with distance (per unit). |
| `category_bonus` | float | 0.05 | 0.01-0.2 | Additional bonus for same-category adjacency. |

### Popularity Placement Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `high_traffic_multiplier` | float | 1.2 | 1.0-2.0 | Bonus multiplier for popular items in high-traffic areas. |
| `popularity_threshold` | float | 0.3 | 0.1-0.8 | Support threshold for "popular" classification. |
| `endcap_bonus` | float | 1.5 | 1.0-3.0 | Additional bonus for popular items in premium endcap positions. |

### Cross-Selling Optimization

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_confidence` | float | 0.6 | 0.3-0.9 | Minimum confidence for cross-selling consideration. |
| `complementary_bonus` | float | 0.2 | 0.05-0.5 | Bonus for placing complementary products nearby. |
| `basket_affinity_weight` | float | 1.0 | 0.5-2.0 | Weight for products often bought together. |

## Scoring Algorithm Parameters

### Combo Effectiveness Scoring

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `confidence_weight` | float | 0.4 | 0.2-0.6 | Weight of confidence in combo effectiveness calculation. |
| `lift_weight` | float | 0.3 | 0.2-0.5 | Weight of lift in combo effectiveness calculation. |
| `frequency_weight` | float | 0.2 | 0.1-0.4 | Weight of frequency in combo effectiveness calculation. |
| `diversity_weight` | float | 0.1 | 0.05-0.3 | Weight of category diversity in combo effectiveness calculation. |

### Layout Performance Scoring

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `utilization_optimal` | float | 0.85 | 0.7-0.95 | Optimal utilization rate for scoring calculations. |
| `traffic_weight` | float | 0.3 | 0.1-0.6 | Weight of foot traffic in section performance scoring. |
| `cross_sell_weight` | float | 0.4 | 0.2-0.7 | Weight of cross-selling potential in performance scoring. |
| `accessibility_weight` | float | 0.3 | 0.1-0.5 | Weight of accessibility in performance scoring. |

## Performance Parameters

### Computational Limits

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_optimization_time` | int | 300 | 60-1800 | Maximum time (seconds) for optimization algorithms. |
| `max_memory_mb` | int | 2048 | 512-8192 | Maximum memory usage (MB) for optimization processes. |
| `parallel_workers` | int | 4 | 1-16 | Number of parallel workers for optimization tasks. |
| `batch_size` | int | 1000 | 100-5000 | Batch size for processing large datasets. |

### Convergence Criteria

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_improvement_threshold` | float | 0.05 | 0.01-0.2 | Minimum improvement to continue optimization iterations. |
| `max_iterations` | int | 100 | 10-500 | Maximum number of optimization iterations. |
| `convergence_patience` | int | 10 | 3-50 | Number of iterations without improvement before stopping. |

## Parameter Tuning Guidelines

### Store Size Considerations

#### Small Stores (< 1,000 SKUs)
```yaml
association_rules:
  min_support: 0.02
  min_confidence: 0.5
  min_lift: 1.1

optimization:
  target_utilization: 0.9
  rebalance_frequency_days: 3

heuristics:
  crowding_penalty:
    base_penalty: 0.1
    utilization_threshold: 0.95
```

#### Medium Stores (1,000 - 5,000 SKUs)
```yaml
association_rules:
  min_support: 0.03
  min_confidence: 0.6
  min_lift: 1.2

optimization:
  target_utilization: 0.85
  rebalance_frequency_days: 7

heuristics:
  crowding_penalty:
    base_penalty: 0.15
    utilization_threshold: 0.9
```

#### Large Stores (> 5,000 SKUs)
```yaml
association_rules:
  min_support: 0.05
  min_confidence: 0.7
  min_lift: 1.3

optimization:
  target_utilization: 0.8
  rebalance_frequency_days: 14

heuristics:
  crowding_penalty:
    base_penalty: 0.2
    utilization_threshold: 0.85
```

### Business Objective Tuning

#### Revenue Maximization
- Increase `revenue_optimization` weight to 0.6
- Higher `popularity_threshold` (0.4-0.5)
- Aggressive `endcap_bonus` (2.0-3.0)
- Lower `min_confidence` for associations (0.5-0.6)

#### Customer Experience Focus
- Increase `customer_experience` weight to 0.5
- Lower `target_utilization` (0.8)
- Higher `accessibility_weight` (0.4-0.5)
- Stricter `crowding_penalty` settings

#### Operational Efficiency
- Increase `operational_efficiency` weight to 0.4
- Higher `target_utilization` (0.9)
- Longer `rebalance_frequency_days` (14-21)
- Lower `max_iterations` for faster optimization

## Configuration Examples

### High-Performance Configuration
```yaml
# Optimized for stores with >10k SKUs requiring <180s optimization time
association_rules:
  min_support: 0.08
  min_confidence: 0.75
  min_lift: 1.4
  max_itemset_size: 3

performance:
  max_optimization_time: 180
  parallel_workers: 8
  batch_size: 2000
  max_iterations: 50

heuristics:
  crowding_penalty:
    enabled: true
    base_penalty: 0.2
    exponential_scaling: 2.5
  
  adjacency_reward:
    enabled: true
    association_threshold: 1.8
    distance_decay: 0.7
```

### Accuracy-Focused Configuration
```yaml
# Optimized for maximum accuracy, longer computation time acceptable
association_rules:
  min_support: 0.01
  min_confidence: 0.8
  min_lift: 1.2
  max_itemset_size: 5

performance:
  max_optimization_time: 600
  parallel_workers: 4
  batch_size: 500
  max_iterations: 200
  convergence_patience: 20

heuristics:
  crowding_penalty:
    enabled: true
    base_penalty: 0.1
    exponential_scaling: 1.5
  
  cross_selling:
    min_confidence: 0.7
    complementary_bonus: 0.3
```

### Development/Testing Configuration
```yaml
# Fast configuration for development and testing
association_rules:
  min_support: 0.1
  min_confidence: 0.6
  min_lift: 1.3
  max_itemset_size: 2

performance:
  max_optimization_time: 30
  parallel_workers: 2
  batch_size: 500
  max_iterations: 20
  convergence_patience: 5

heuristics:
  crowding_penalty:
    enabled: false
  
  adjacency_reward:
    enabled: false
```

## Parameter Validation

### Automatic Validation Rules

The system automatically validates parameters and applies constraints:

1. **Range Checking**: All parameters are checked against their valid ranges
2. **Sum Constraints**: Objective weights are normalized to sum to 1.0
3. **Logical Constraints**: E.g., `min_confidence` cannot exceed `max_confidence`
4. **Performance Constraints**: Parameters that would exceed memory/time limits are adjusted

### Manual Validation Checklist

Before deploying new parameters:

- [ ] Association rule parameters produce reasonable number of rules (100-10,000)
- [ ] Optimization completes within target time limits
- [ ] Memory usage stays within system limits
- [ ] Generated recommendations are sensible for your business context
- [ ] Performance metrics meet your quality standards

## Monitoring and Adjustment

### Key Metrics to Monitor

1. **Optimization Runtime**: Track against target thresholds
2. **Rule Quality**: Monitor confidence and lift distributions
3. **Business Metrics**: Sales lift, customer satisfaction
4. **System Resources**: Memory and CPU usage

### Automated Parameter Adjustment

The system can automatically adjust parameters based on performance:

```python
# Example of adaptive parameter adjustment
if optimization_time > target_time:
    min_support *= 1.2  # Reduce rule generation
    max_iterations *= 0.8  # Limit iterations
    
if rule_count < minimum_rules:
    min_support *= 0.9  # Generate more rules
    min_confidence *= 0.95  # Be slightly less strict
```

## Conclusion

Proper parameter tuning is essential for optimal performance of the supermarket layout optimizer. Start with the recommended defaults for your store size and business objectives, then iteratively refine based on performance monitoring and business results.

For additional assistance with parameter tuning, consult the [Performance Benchmarking Guide](benchmarking.md) and [Troubleshooting Guide](troubleshooting.md). 