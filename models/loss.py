import torch
import torch.nn.functional as F


def cross_modal_contrastive_loss(z1, z2, temperature=0.07):
    """
    Contrastive loss between two batches of embeddings, z1 and z2.
    We treat (z1[i], z2[i]) as the positive pair, and all others as negatives.
    """
    # 1) L2-normalize each embedding
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    # 2) Similarity matrix: [batch_size, batch_size]
    # each entry sim[i, j] = dot(z1[i], z2[j]) / temperature
    sim = torch.matmul(z1, z2.t()) / temperature

    # 3) For row i, the correct "label" is i (the diagonal)
    labels = torch.arange(batch_size, device=z1.device)

    # 4) Cross entropy loss
    # We'll interpret each row i of 'sim' as a distribution over j,
    # and the "correct" j is i.
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)
    loss = 0.5 * (loss_12 + loss_21)

    return loss
