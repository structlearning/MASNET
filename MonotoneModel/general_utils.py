import torch

def predict(f_S1, f_S2,eps = 1e-6):
    """
    Predicts labels based on coordinate-wise comparison of f(S1) and f(S2) using PyTorch.

    Parameters:
    - f_S1: torch tensor of shape (batch_size, num_channels) representing f(S1)
    - f_S2: torch tensor of shape (batch_size, num_channels) representing f(S2)
    - delta: threshold for weak dominance condition (scalar)

    Returns:
    - A tensor of shape (batch_size,) with values:
        - 0 if f(S1) <= f(S2) and exists i such that f(S1)_i + delta <= f(S2)_i
        - 1 otherwise
    """
    # Ensure tensors are of float type
    # f_S1, f_S2 = f_S1.float(), f_S2.float()

    # Check if f(S1) <= f(S2) coordinate-wise (for each batch)
    cond_1 = torch.all(f_S1 <= f_S2+eps, dim=1)

    # Assign label 0 where the conditions hold
    result = torch.ones(f_S1.shape[0], dtype=torch.long, device=f_S1.device)  # Default to 1
    result[cond_1] = 0  # Set to 0 if conditions hold

    return result

def set_loss(f_S1, f_S2, delta, y):
    """
    Computes the loss function:
        L(f(S1), f(S2)) = max(0, min_i [f(S2)_i - f(S1)_i + delta])
    
    Args:
        f_S1 (torch.Tensor): Tensor of shape (batch_size, d), representation of S1.
        f_S2 (torch.Tensor): Tensor of shape (batch_size, d), representation of S2.
        delta (float): Margin delta, default is 0.1.
    
    Returns:
        torch.Tensor: Loss value for each batch element.
    """
    # Compute min_i [f(S2)_i - f(S1)_i + delta] for each sample in the batch
    # We can try also with Sum.
    loss1 = torch.clamp(delta + f_S2 - f_S1, min=0)
    loss_1 = loss1.min(dim=1)[0]
    # We can try sum or max or min.
    # Compute loss for batch[i] = 0 (using (S1 subset S2))
    loss2 = torch.clamp(delta + f_S1 - f_S2, min=0)
    loss_2 = loss2.max(dim=1)[0]
        
    # Compute final loss using batch mask   
    loss = (y) * loss_1 + (1-y) * loss_2

    # Return sum over batch
    return loss.mean()