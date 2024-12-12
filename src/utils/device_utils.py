import torch

def get_device(device_name='auto'):
    """获取计算设备"""
    if device_name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_name) 