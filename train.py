import argparse

from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

from models import get_model


def main(opt):
    exp = Experiment(save_dir=f'./logs/{opt.model}')

    exp.argparse(opt)
    exp.tag({'description': opt.model})

    model = get_model(opt)

    trainer = Trainer(experiment=exp,
                      max_nb_epochs=4000,
                      check_val_every_n_epoch=10,
                      gpus=[0])

    trainer.fit(model)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--model', choices=['srcnn', 'srgan'], required=True)
    args.add_argument('--dataroot', type=str, default='./data/DIV2K')
    args.add_argument('--scale_factor', type=int, default=4)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--gpus', )
    opt = args.parse_args()

    main(opt)