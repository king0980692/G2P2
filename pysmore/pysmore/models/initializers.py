import random
import torch


class RandomInitializer:
    """
    mimic the SMORe's init embedding
    """

    def __new__(self, vocab_size, emb_dim):
        RAND_MAX = 2 ^ 31
        vocab_size = vocab_size
        emb_dim = emb_dim

        embed = []
        for i in range(vocab_size):
            tmp_emb = []
            for j in range(emb_dim):
                r_num = ((random.randint(0, RAND_MAX) / RAND_MAX) - 0.5) / emb_dim
                tmp_emb.append(r_num)
            embed.append(tmp_emb)

        return torch.tensor(embed, requires_grad=True)


def random_initializer(emb_weight, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    vocab_size, emb_dim = emb_weight.size()
    # 使用 torch.rand 生成随机数张量
    new_tensor = torch.rand(vocab_size, emb_dim)

    with torch.no_grad():  # 确保更新操作不会影响梯度计算
        emb_weight.data.copy_(new_tensor)


def smore_initializer(emb_weight, seed=None):
    """
        Mimic the SMORe's init embedding
    """

    if seed is not None:
        random.seed(seed)

    RAND_MAX = 2 ** 31
    vocab_size, emb_dim = emb_weight.size()

    new_tensor = torch.empty_like(emb_weight)
    for i in range(vocab_size):
        for j in range(emb_dim):
            r_num = (random.randint(0, RAND_MAX) / RAND_MAX - 0.5) / emb_dim
            new_tensor[i, j] = r_num

    with torch.no_grad():
        emb_weight.data.copy_(new_tensor)
