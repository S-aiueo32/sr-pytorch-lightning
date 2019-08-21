from .srcnn_model import SRCNNModel
from .srgan_model import SRGANModel

def get_model(opt):
    if opt.model == 'srcnn':
        return SRCNNModel(opt)
    elif opt.model == 'srgan':
        return SRGANModel(opt)
    else:
        NotImplementedError