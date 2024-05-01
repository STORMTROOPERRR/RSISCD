from torch.optim import lr_scheduler


def get_scheduler(optimizer, args):
    if args['lr_strategy'] == 'default':
        scheduler = None

    elif args['lr_strategy'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=0.9)

    elif args['lr_strategy'] == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    elif args['lr_strategy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['T_max'], eta_min=0.0005)

    elif args['lr_strategy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=0.0005,
                                                   patience=args['patience'])

    elif args['lr_strategy'] == 'MultiplicativeLR':
        scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)

    elif args['lr_strategy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args['epochs'] + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif args['lr_strategy'] == 'LinearLR':
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.005, total_iters=args['iters'])

    else:
        return NotImplementedError(f"learning rate policy {args['lr_strategy']} is not implemented")
    return scheduler
