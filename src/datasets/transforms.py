import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from functools import partial

def get_transform(transform_type):
    if transform_type == 'none':
        forward = transforms.Compose([partial(TF.convert_image_dtype)])
        T128 = forward
        T32 = T128
        T16 = T128
        T8  = T128
        T4  = T128
        
    elif transform_type == 'ill':
        bf, bi = RandomBrightness(0.75, 1.25)
        forward = transforms.Compose([bf])
        T128 = transforms.Compose([bi])
        T32 = transforms.Compose([partial(TF.convert_image_dtype)])
        T16 = T32
        T8  = T32
        T4  = T32
    
    elif transform_type == 'trans':
        tf, ti_128, ti_32, ti_16, ti_8, ti_4 = RandomTranslation(-10,11)
        forward = transforms.Compose([tf])
        T128 = transforms.Compose([ti_128,])
        T32  = transforms.Compose([ti_32, ])
        T16  = transforms.Compose([ti_16, ])
        T8   = transforms.Compose([ti_8,  ])
        T4   = transforms.Compose([ti_4,  ])
    
    elif transform_type == 'rot':
        rf, ri = RandomRotation(15,346)
        forward = transforms.Compose([rf])
        T128 = transforms.Compose([ri])
        T32 = T128
        T16 = T128
        T8  = T128
        T4  = T128
        
    elif transform_type == 'all':
        bf, bi = RandomBrightness(0.75, 1.25)
        rf, ri = RandomRotation(15,346)
        tf, ti_128, ti_32, ti_16, ti_8, ti_4 = RandomTranslation(-10,11)
        forward = transforms.Compose([bf, rf, tf])
        
        T128 = transforms.Compose([ti_128, ri])
        T32  = transforms.Compose([ti_32,  ri])
        T16  = transforms.Compose([ti_16,  ri])
        T8   = transforms.Compose([ti_8,   ri])
        T4   = transforms.Compose([ti_4,   ri])
        
    return forward, T128, T32, T16, T8, T4

def normalize_image(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

def random_brightness(x, low, high):
    val = (low - high) * torch.rand(1) + high
    return TF.adjust_brightness(x, val), val
    
def RandomRotation(low, high):
    val = torch.randint(low, high, (1,))
    forward = transforms.Lambda(lambda x: TF.rotate(x, val.item()))
    inverse = partial(TF.rotate, angle=-val.item())
    return forward, inverse

def random_rotation(x, low, high):
    val = torch.randint(low, high, (1,))
    return TF.rotate(x, val.item()), val

def RandomBrightness(low, high):
    val = torch.FloatTensor(1).uniform_(low, high)
    v_diff = 1.-val.item()
    forward = transforms.Lambda(lambda x: TF.adjust_brightness(x, val.item()))
    invsere = partial(TF.adjust_brightness, brightness_factor=1+v_diff)
    return forward, invsere

def TranslationRatio(x,y, size=128, new_size=32):
    new_x = new_size*(x/size)
    new_y = new_size*(y/size)
    return new_x, new_y

def RandomTranslation(low, high):
    xt = torch.randint(low,high,(1,)).item()
    yt = torch.randint(low,high,(1,)).item()
    f = transforms.Lambda(lambda x: TF.affine(x, 0, [xt,yt], 1, 0))
    i_128 = partial(TF.affine, angle=0, translate=[-xt,-yt], scale=1, shear=0)
    
    # Translation inverse transforms for lower scale maps
    x32, y32 = TranslationRatio(xt,yt, size=128, new_size=32)
    x16, y16 = TranslationRatio(xt,yt, size=128, new_size=16)
    x8, y8 = TranslationRatio(xt,yt, size=128, new_size=8)
    x4, y4 = TranslationRatio(xt,yt, size=128, new_size=4)
    
    i_32 = partial(TF.affine, angle=0, translate=[-x32,-y32], scale=1, shear=0)
    i_16 = partial(TF.affine, angle=0, translate=[-x16,-y16], scale=1, shear=0)
    i_8 = partial(TF.affine, angle=0, translate=[-x8,-y8], scale=1, shear=0)
    i_4 = partial(TF.affine, angle=0, translate=[-x4,-y4], scale=1, shear=0)
    
    return f, i_128, i_32, i_16, i_8, i_4

def random_translation(x, low, high):
    xt = torch.randint(low,high, (1,))
    yt = torch.randint(low,high, (1,))
    neg = torch.rand((2))
    negx,negy = neg
    if negx >= 0.5:
        xt = -xt
    if negy >= 0.5:
        yt = -yt
    x = TF.affine(x, 0, [xt,yt], 1, 0)
    return x, xt, yt

def InverseData(t_string):
    if t_string == 'none':
        # Dummy op
        return partial(TF.convert_image_dtype)
    elif t_string == 'fv' or t_string == 'vf':
        return partial(TF.vflip)
    elif t_string == 'fh' or t_string == 'hf':
        return partial(TF.hflip)
    elif 'r' in t_string:
        ang = int(t_string.split('r')[-1])
        return partial(TF.rotate, angle=ang)