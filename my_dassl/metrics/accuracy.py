from torch.nn import functional as F

def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res

def compute_cosine_similarity(original, reconstructed):
    """
    Computes the average cosine similarity between the original and reconstructed tensors.

    Args:
        original (torch.Tensor): The original tensor with shape (batch_size, C, H, W) where
            - batch_size: the number of samples in the batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
        reconstructed (torch.Tensor): The reconstructed tensor with the same shape as the original tensor.
        
    Returns:
        float: The average cosine similarity between all pairs of original and reconstructed samples in the batch.
    """
    
    # Reshape the original and reconstructed tensors into 2D tensors where each row is a sample
    original_flat = original.view(original.size(0), -1)
    reconstructed_flat = reconstructed.view(reconstructed.size(0), -1)
    
    # Compute the cosine similarity between each pair of original and reconstructed samples
    similarity = F.cosine_similarity(original_flat, reconstructed_flat)
    
    # Compute and return the mean cosine similarity
    return similarity.mean().item()