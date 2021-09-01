from network.aquanet import Aquanet
from network.pspnet import PSPNet
from network.emanet import EMANet
from network.ocrnet import OCRNet
from network.ocnet import OCNet
from network.deeplabv3 import DeepLabV3
from network.ccnet import CCNet
from network.danet import DANet
from network.annet import ANNet
from network.gcnet import GCNet
from network.dnlnet import DNLNet
import torch

def models(args):
    if args.model == 'AquaNet':
        model = Aquanet(num_classes=args.num_classes)
    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes)
    if args.model == 'DANet':
        model = DANet(num_classes=args.num_classes)
    if args.model == 'ANNet':
        model = ANNet(num_classes=args.num_classes)
    if args.model == 'EMANet':
        model = EMANet(num_classes=args.num_classes)
    if args.model == 'CCNet':
        model = CCNet(num_classes=args.num_classes)
    if args.model == 'DeepLabV3':
        model = DeepLabV3(num_classes=args.num_classes)
    if args.model == 'OCRNet':
        model = OCRNet(num_classes=args.num_classes)
    if args.model == 'OCNet':
        model = OCNet(num_classes=args.num_classes)
    if args.model == 'GCNet':
        model = GCNet(num_classes=args.num_classes)
    if args.model == 'DNLNet':
        model = DNLNet(num_classes=args.num_classes)
    return model

def build_model(args):
    model = models(args)

    saved_state_dict = torch.load(args.restore_from)
    new_params = model.backbone.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.backbone.load_state_dict(new_params)

    return model


def load_model(args):
    model = models(args)

    saved_state_dict = torch.load(args.restore_from)
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    return model