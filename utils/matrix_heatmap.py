import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def save_matrix_heatmap(matrix, filename, title="Matrix Visualization"):
    """
    Saves a heatmap visualization of a matrix.
    Safely handles PyTorch tensors on GPUs.
    """
    # 1. Safely convert to numpy array if it's a PyTorch tensor
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
        
    # 2. Set up the figure
    plt.figure(figsize=(8, 8))
    
    # 3. Center the colormap at exactly 0.0
    # This makes 0 white, positive values red, and negative values blue.
    abs_max = np.max(np.abs(matrix))
    # Add a tiny epsilon to prevent division by zero if the matrix is completely empty
    vmax = abs_max if abs_max > 0 else 1e-5
    vmin = -vmax
    
    # 4. Draw the heatmap
    # 'nearest' interpolation prevents blurring so you can see exact structural sparsity
    img = plt.imshow(matrix, cmap='coolwarm', aspect='auto', 
                     interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # 5. Add colorbar and labels
    plt.colorbar(img, fraction=0.046, pad=0.04, label='Weight Value')
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Rows', fontsize=12)
    
    # 6. Save and close
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {filename}")