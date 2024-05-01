# -*- coding: utf-8 -*-
"""
# Created on 14:45:07 2024/5/1

"""
from torch.utils.data import DataLoader


def get_dataloader(data_name, train_batch_size, val_batch_size):
    if data_name == 'SECOND':
        from dataset import SECOND as DATA_PROCESSOR
    elif data_name == 'Landsat':
        from dataset import Landsat_SCD as DATA_PROCESSOR
    else:
        raise NotImplementedError(f"{data_name} not supported!")
    num_classes = DATA_PROCESSOR.num_classes
    train_set = DATA_PROCESSOR.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=4, shuffle=True)
    val_set = DATA_PROCESSOR.Data('val')
    val_loader = DataLoader(val_set, batch_size=val_batch_size, num_workers=4, shuffle=False)
    return train_loader, val_loader, num_classes
