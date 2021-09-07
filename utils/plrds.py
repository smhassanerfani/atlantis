
# Polynomial Learning Rate Decay Scheduler 

def lr_poly(base_lr, i_iter, max_iter, power):
    return base_lr * ((1 - float(i_iter) / max_iter) ** (power))

def adjust_learning_rate(args, optimizer, i_iter, max_iter):
    lr = lr_poly(args.learning_rate, i_iter, max_iter, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr
