import torch.distributed as dist

# Weighted gradient averaging
def average_gradients_weighted(model, world_size, local_weight):
    for param in model.parameters():
        if param.grad is not None:
            original_grad = param.grad.data.clone()           
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)            
            param.grad.data = local_weight * original_grad + \
                              (1 - local_weight) * (param.grad.data - original_grad) / (world_size - 1)
            
# Weighted model weights averaging
def average_weights_weighted(model, world_size, local_weight):
    for param in model.parameters():
        original_grad = param.data.clone()
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data = local_weight * original_grad + \
                            (1 - local_weight) * (param.data - original_grad) / (world_size - 1)

# Equal gradient averaging
def average_gradients_equal(model, world_size):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

# Equal model weights averaging
def average_weights_equal(model, world_size):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size