# CFNOW

The simplest way to generate counterfactuals for any tabular dataset and model.

This package finds an optimal point (closer to the  input dataset point), which the classification is different from the original classification (i.e. "flips" the classification of the original input by minimally changin it).

Minimal example:
```python
import cfnow

# Generating a sample model

# Selecting a random point

# Here we can see the original class

# Then, we use CFNOW to generate the minimum modification to change the classification

```

## Improving your results
The minimal example above considers all features as numerical continuous, however, some datasets can have categorical (binary or one-hot encoded) features. CFNOW can handle these data types in a simple way as demonstrated below:

### I have binary categorical features!
```python
import cfnow

# Generating a sample model



```

### I have one-hot encoded features!
```python
import cfnow

# Generating a sample model



```

### I have one-hot and binary categorical features!
```python
import cfnow

# Generating a sample model



```