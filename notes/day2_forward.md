# Day 2: Forward Propagation Theory

## 1. What is Forward Propagation?
Forward propagation is the process where input data passes through a neural network layer by layer to produce an output. It's the "forward" flow of information from input → hidden layers → output.

## 2. The Neuron: Basic Building Block
A single neuron performs two operations:
1. **Linear Transformation**: Weighted sum of inputs plus bias
2. **Non-linear Activation**: Apply activation function

### Mathematical Representation:
For a single neuron with `n` inputs:

z = (w₁ * x₁) + (w₂ * x₂) + ... + (wₙ * xₙ) + b

a = activation_function(z)   # z as input


Where:
- `xᵢ` = input features
- `wᵢ` = weights (parameters to learn)
- `b` = bias (parameter to learn ,some constant added to the weighted sum to make it non-zero)
- `z` = weighted sum (pre-activation)
- `a` = activation (neuron output)

## 3. Layer-wise Forward Propagation
For a layer with multiple neurons:

### Matrix Formulation:

Z = W · X + b

A = activation(Z)


Where:
- `X` = input matrix (shape: features × batch_size)
- `W` = weight matrix (shape: neurons_in_current_layer × input_features)
- `b` = bias vector (shape: neurons_in_current_layer × 1)
- `Z` = pre-activation matrix
- `A` = activation output

## 4. Complete Forward Pass Through Network
For a 3-layer MLP (Input → Hidden → Output):
```
X = input

Hidden layer
Z₁ = W₁ · X + b₁
A₁ = ReLU(Z₁)

Output layer
Z₂ = W₂ · A₁ + b₂
A₂ = Sigmoid(Z₂)  # or other output activation

Output = A₂

```
## 5. Activation Functions
Why we need activation functions:
- Introduce non-linearity (otherwise network is just linear)
- Enable learning complex patterns
- Different functions for different layers

### Common Activation Functions:

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary classification output |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers |
| **Softmax** | eˣⁱ/∑eˣʲ | (0, 1) | Multi-class output |

## 6. Numerical Example
Let's trace a single sample through a small network:
```
**Input**: x = [2.0, 3.0]
**Weights Hidden Layer** (2 inputs → 3 neurons):

W₁ = [[0.1, 0.2], # Neuron 1 weights

[0.3, 0.4], # Neuron 2 weights

[0.5, 0.6]] # Neuron 3 weights

b₁ = [0.1, 0.2, 0.3]
```

**Calculation**:
```

Z₁ = W₁ · x + b₁

= [[0.1*2 + 0.2*3 + 0.1],

[0.3*2 + 0.4*3 + 0.2],

[0.5*2 + 0.6*3 + 0.3]]

= [0.9, 2.0, 3.1]

A₁ = ReLU(Z₁) = [0.9, 2.0, 3.1] (all positive)
```

## 7. Key Points to Remember
1. **No Learning Happens**: Forward propagation doesn't update weights
2. **Deterministic**: Same input → same output (no randomness in forward pass)
3. **Batch Processing**: Can process multiple samples simultaneously using matrix operations
4. **Sequential**: Each layer's output becomes next layer's input
5. **Efficiency**: Matrix operations are highly optimized on GPU

## 8. Forward Propagation Steps Summary
1. Start with input data
2. For each layer:
   - Compute linear transformation: z = w·x + b
   - Apply activation function: a = f(z)
   - Pass output to next layer
3. Final layer produces network output
4. Compare output with target (for loss calculation in backprop, error calculation)

## 9. Visual Representation
```
Input Layer Hidden Layer Output Layer
[x₁]        [h₁]         [y]
[x₂] →      [h₂] →
            [h₃]

Where:
hᵢ = ReLU(w₁ᵢx₁ + w₂ᵢx₂ + bᵢ)
y = σ(w₁h₁ + w₂h₂ + w₃*h₃ + b)

```
This completes the forward pass. 
