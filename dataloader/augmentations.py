import numpy as np
import torch
import random


def DataTransform_time(time):

    # augmentation for time
    time_augs = [scaling, jitter, masking]
    chosen_time_aug = random.choice(time_augs)
    time_aug_result = chosen_time_aug(time)

    return time_aug_result



def jitter(x, sigma=2): #for time input
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.5): #for time input
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    # return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)
    return np.random.binomial(1, p, size=(B, T, D)).astype(bool)

def masking(x, drop_ratio=0.2, mask= 'binomial'):
    global mask_id
    # nan_mask = ~x.isnan().any(axis=-1)
    nan_mask = ~np.isnan(x).any(axis=-1)
    x[~nan_mask] = 0

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.shape[0], x.shape[1], x.shape[2], p=1-drop_ratio)

    # mask &= nan_mask
    x[~mask_id] = 0
    return x


