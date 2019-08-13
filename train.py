from pytorch_lightning.models.trainer import Trainer

from model import SRCNNModel


def main():
    model = SRCNNModel()
    trainer = Trainer()
    trainer.fit(model)

if __name__ == "__main__":
    main()