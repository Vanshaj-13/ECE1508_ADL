import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    """Derivative of Sigmoid: s * (1 - s)"""
    s = sigmoid(x)
    return s * (1 - s)

def d_tanh(x):
    """Derivative of Tanh: 1 - tanh^2"""
    return 1 - np.tanh(x)**2

# 1. Setup Parameters
epsilon = 0.05
T_values = np.arange(1, 21)  # T ranges from 1 to 20

# 2. Calculate Ratios
# Approximative Ratio: derived from Maclaurin series at 0
# d_sigmoid(0) = 0.25, d_tanh(0) = 1.0
rho_approx = 0.25 / 1.0 

# True Ratio: using actual derivative values at epsilon
# Note: Assuming the question implies ratio of derivatives based on context
val_d_sigmoid = d_sigmoid(epsilon)
val_d_tanh = d_tanh(epsilon)
rho_true = val_d_sigmoid / val_d_tanh

print(f"Approximative Ratio: {rho_approx}")
print(f"True Ratio at epsilon={epsilon}: {rho_true:.4f}")

# 3. Calculate Curves (Ratio ^ T)
y_approx = rho_approx ** T_values
y_true = rho_true ** T_values

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(T_values, y_approx, 'b--o', label=r'Approximative: $(0.25)^T$')
plt.plot(T_values, y_true, 'r-x', label=r'True Ratio: $(\frac{\dot{\sigma}(\epsilon)}{\dot{\tanh}(\epsilon)})^T$')

plt.title(f'Vanishing Gradient: Decay of Gradient Ratio over Time (epsilon={epsilon})')
plt.xlabel('Time steps (T)')
plt.ylabel(r'Ratio $\rho^T$')
plt.xticks(T_values)  # Show all integer ticks for T
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.yscale('linear') # Keep linear to show the rapid drop to zero

# Save the plot
plt.savefig('gradient_decay_plot.png')
plt.show()