import numpy as np
from PIL import Image

palette = [0,0,0,128,0,0,0,128,0,128,128,0,0,0,128,128,0,128,0,128,128,128,128,128,64,0,0,192,0,0,64,128,0,192,128,0,
           64,0,128,192,0,128,64,128,128,192,128,128,0,64,0,128,64,0,0,192,0,128,192,0,0,64,128,128,64,128,0,192,128,
           128,192,128,64,64,0,192,64,0,64,192,0,192,192,0,64,64,128,192,64,128,64,192,128,192,192,128,0,0,64,128,0,
           64,0,128,64,128,128,64,0,0,192,128,0,192,0,128,192,128,128,192,64,0,64,192,0,64,64,128,64,192,128,64,64,0,
           192,192,0,192,64,128,192,192,128,192,0,64,64,128,64,64,0,192,64,128,192,64,0,64,192,128,64,192,0,192,192,
           128,192,192,64,64,64]

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(args, optimizer, i_iter, total_steps):
    lr = lr_poly(args.learning_rate, i_iter, total_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask