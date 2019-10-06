def get_weight_norm(model, norm_type=2):
    """Get the norm of the model weights"""
    norm = 0
    for p in model.parameters():
        norm += p.data.norm(norm_type).item() ** norm_type
    return norm ** (1. / norm_type)

def get_grad_norm(model, norm_type=2):
    """Get the norm of the model weight gradients"""
    norm = 0
    for p in model.parameters():
        norm += p.grad.data.norm(norm_type).item() ** norm_type
    return norm ** (1. / norm_type)
