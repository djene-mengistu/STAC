import sys
import importlib
import argparse
import torch


def get_model(args):
    method = getattr(importlib.import_module(args.network), 'Net')
    if args.network_type == 'eps':
        # model = method(args.num_classes + 1)
        model = method(args.num_classes)
    else:
        model = method(args.num_classes)

    if args.weights[-7:] == '.params':
        assert args.network in ["network.resnet38_cls", "network.resnet38_eps"]
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    return model


# model = get_model(args)
# # print(model.)

# img = torch.randn(2,3,224,224)
# y, z = model(img)
# print(y.shape, z.shape)