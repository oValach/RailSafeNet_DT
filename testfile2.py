import torch

# Create a sample tensor
tensor = torch.rand(4, 21, 224, 224)  # Replace with your actual tensor

# Switch dimensions 0 and 1
switched_tensor = tensor.permute(1, 0, 2, 3)
print('')