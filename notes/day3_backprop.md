### Day 3: Backpropagation - Complete Theory Guide

## 1. Why Backpropagation?
Backpropagation is the algorithm used to compute how much each parameter (weight or bias) contributed to the final loss. This allows the model to:

Know which direction to adjust the weights.
Reduce the loss over time.
Learn from data.

**In simple terms:**
**Backpropagation = calculating gradients to improve the model.**

---

## 2. Training Recipe (PyTorch Workflow)
Every training step follows this exact sequence:

### **Training Loop Steps:**

1. **Forward pass**
   ```python
   predictions = model(inputs)
   ```

2. **Compute loss**
   ```python
   loss = loss_function(predictions, targets)
   ```

3. **Zero previous gradients**
   ```python
   optimizer.zero_grad()
   ```

4. **Backward pass (compute gradients)**
   ```python
   loss.backward()
   ```

5. **Update parameters**
   ```python
   optimizer.step()
   ```

---

## 3. Chain Rule for a Single Neuron

### **Forward Equations:**

#### **Linear step:**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

#### **Activation:**
```
a = σ(z)   where σ = activation function
```

#### **Loss (Mean Squared Error for one sample):**
```
L = ½(a - y)²
```

### **Step-by-step Derivatives (Backward Pass):**

#### **1. Loss with respect to output:**
```
∂L/∂a = (a - y)
```

#### **2. Output with respect to z (depends on activation):**
```
∂a/∂z = derivative of activation function
```

#### **3. Combine using chain rule:**
```
∂L/∂z = (∂L/∂a) × (∂a/∂z)
```

#### **4. Gradient with respect to weight wᵢ:**
```
∂L/∂wᵢ = (∂L/∂z) × xᵢ
```

#### **5. Gradient with respect to bias b:**
```
∂L/∂b = (∂L/∂z) × 1
```

---

## 4. Weight Update Rule (Gradient Descent)

After computing gradients:

### **Update Equations:**
```
w_new = w_old − η × (∂L/∂w)
b_new = b_old − η × (∂L/∂b)
```

Where:
- **η (eta)** = learning rate
- Controls how big each update step is

### **Learning Rate Importance:**
- **Too small**: Slow learning, may get stuck
- **Too large**: Overshoot, may diverge
- **Just right**: Smooth convergence

---

## 5. Activation Function Derivatives

### **Sigmoid Derivative:**
```
σ(z) = 1 / (1 + e⁻ᶻ)
σ'(z) = σ(z) × (1 - σ(z))
```

### **ReLU Derivative:**
```
ReLU(z) = max(0, z)
ReLU'(z) = 
    1  if z > 0
    0  if z < 0
```

**Note:** At z = 0, derivative is undefined. Usually treated as 0 or 1 depending on implementation.

### **Tanh Derivative:**
```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
tanh'(z) = 1 - tanh²(z)
```

---

## 6. Complete Example: Backpropagation Through a Simple Network

### **Network Architecture:**
```
Input(x) → Weight(w) → Sigmoid → Output(a) → Loss(L)
```

### **Forward Pass:**
```
z = w × x
a = σ(z) = 1/(1+e⁻ᶻ)
L = ½(a - y)²
```

### **Backward Pass:**
```
∂L/∂a = a - y
∂a/∂z = a × (1 - a)
∂L/∂z = (a - y) × a × (1 - a)
∂L/∂w = (a - y) × a × (1 - a) × x
```

### **Update:**
```
w_new = w_old - η × ∂L/∂w
```

---

## 7. Visualizing Gradient Flow

### **Positive Gradient (∂L/∂w > 0):**
- Loss increases as weight increases
- Need to **decrease** weight

### **Negative Gradient (∂L/∂w < 0):**
- Loss decreases as weight increases
- Need to **increase** weight

### **Zero Gradient (∂L/∂w = 0):**
- At minimum (or flat region)
- No update needed

---

## 8. Practical Training Tips

### **1. Always zero gradients:**
```python
optimizer.zero_grad()  # MUST DO THIS BEFORE loss.backward()
```
- Gradients accumulate by default in PyTorch
- Without this, gradients from previous batches add up

### **2. If training diverges or explodes:**
- Reduce the learning rate
- Check for gradient clipping
- Normalize input data

### **3. If gradients become zero (vanishing gradient):**
- Check activation functions
- Use ReLU instead of Sigmoid/Tanh for deep networks
- Consider batch normalization

### **4. ReLU neurons can "die":**
- If always negative, gradient = 0
- Neuron stops learning permanently
- Solution: Use Leaky ReLU or proper initialization

---

## 9. Common Loss Functions and Their Gradients

### **Mean Squared Error (MSE):**
```
L = ½(y_pred - y_true)²
∂L/∂y_pred = y_pred - y_true
```

### **Binary Cross Entropy (BCE):**
```
L = -[y_true × log(y_pred) + (1-y_true) × log(1-y_pred)]
∂L/∂y_pred = (y_pred - y_true) / [y_pred × (1-y_pred)]
```

### **Categorical Cross Entropy (with Softmax):**
```
L = -∑ y_true_i × log(y_pred_i)
∂L/∂z_i = y_pred_i - y_true_i  (after softmax)
```

---

## 10. Big Picture: Complete Training Cycle

### **One Training Step:**
```
1. Forward Pass:    Input → Model → Prediction
2. Compute Loss:    Compare Prediction with Target
3. Backward Pass:   Calculate Gradients (loss.backward())
4. Update Weights:  optimizer.step()
```

### **Training Loop Structure:**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get batch
        inputs, targets = batch
        
        # 1. Forward pass
        predictions = model(inputs)
        
        # 2. Compute loss
        loss = criterion(predictions, targets)
        
        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 4. Update weights
        optimizer.step()
```

---

## 11. Key Concepts Summary

### **Forward Propagation:**
- Data flows from input to output
- Computes predictions
- **No learning** happens here

### **Backward Propagation:**
- Error flows from output back to input
- Computes gradients for each parameter
- **Learning** happens here via weight updates

### **Gradient:**
- Tells us which direction to adjust weights
- Magnitude tells us how much to adjust

### **Learning Rate:**
- Controls step size of weight updates
- Critical hyperparameter to tune

---

## 12. Common Issues and Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Exploding Gradients** | Loss becomes NaN/inf | Reduce learning rate, gradient clipping |
| **Vanishing Gradients** | Loss stops decreasing | Use ReLU, batch normalization |
| **Overfitting** | Training loss ↓, Validation loss ↑ | Add dropout, regularization |
| **Underfitting** | Both losses remain high | Increase model complexity, train longer |

---

## 13. Memory Aid: Backpropagation in 5 Steps

1. **Forward**: Compute predictions
2. **Loss**: Calculate error
3. **Backward**: Find who's to blame (gradients)
4. **Update**: Fix the mistakes (adjust weights)
5. **Repeat**: Until good enough

---

## 14. Essential Code Snippet

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```