import numpy as np
from models import aggregators
from models import backbones
import torchvision


# def get_pretrained_torchvision_model(backbone_name):
#     """This function takes the name of a backbone and returns the pretrained model from torchvision.
#     Examples of backbone_name are 'ResNet18' or 'EfficientNet_B0'
#     """
#     try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
#         weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
#         model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
#     except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
#         model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
#     return model

# def get_randomized_torchvision_model(backbone_name):
#     try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
#         weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
#         model = getattr(torchvision.models, backbone_name.lower())(weights=None)    # 随机初始化
#     except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
#         model = getattr(torchvision.models, backbone_name.lower())(pretrained=False)
#     return model

def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 num_trainable_blocks = 2,
                 ):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where 
                                         we sometimes need to crop the last 
                                         residual block (ex. [4]). Defaults to [].

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        return backbones.ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
    
    elif 'efficientnet_v2' in backbone_arch.lower():
        return backbones.EfficientNet_V2(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

    elif 'efficient' in backbone_arch.lower():
        if '_b' in backbone_arch.lower():
            return backbones.EfficientNet(backbone_arch, pretrained, layers_to_freeze+2)
        else:
            return backbones.EfficientNet(model_name='efficientnet_b0',
                                          pretrained=pretrained, 
                                          layers_to_freeze=layers_to_freeze)
            
    elif 'swin' in backbone_arch.lower():
        return backbones.Swin(model_name='swinv2_base_window12to16_192to256_22kft1k', 
                              pretrained=pretrained, 
                              layers_to_freeze=layers_to_freeze)
    elif 'dino' in backbone_arch.lower():
        return backbones.DINOv2(model_name=backbone_arch, num_trainable_blocks=num_trainable_blocks)
    
    
    # if 'dinov2' in backbone_arch.lower():
    #     return backbones.DINOv2(model_name=backbone_arch, **backbone_config)

def get_aggregator(agg_arch='mixvpr', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)