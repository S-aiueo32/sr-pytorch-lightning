import argparse
import warnings

from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

import models

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['srcnn', 'srgan'], required=True)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--gpus', type=str, default='0')
    opt = parser.parse_args()

    # load model class
    if opt.model == 'srcnn':
        Model = models.SRCNNModel
    elif opt.model == 'srgan':
        Model = models.SRGANModel

    # add model specific arguments to original parser
    parser = Model.add_model_specific_args(parser)
    opt = parser.parse_args()

    # instantiate experiment
    exp = Experiment(save_dir=f'./logs/{opt.model}')
    exp.argparse(opt)

    model = Model(opt)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=exp.get_media_path(exp.name, exp.version),
    )

    # instantiate trainer
    trainer = Trainer(
        experiment=exp,
        max_nb_epochs=4000,
        add_log_row_interval=50,
        check_val_every_n_epoch=10,
        checkpoint_callback=checkpoint_callback,
        gpus=[int(i) for i in opt.gpus.split(',')]
    )

    # start training!
    trainer.fit(model)


if __name__ == "__main__":
    main()
