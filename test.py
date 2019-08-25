import argparse
from pathlib import Path

from kornia.color import rgb_to_grayscale
from kornia.losses import SSIM
from torchvision.utils import save_image
from tqdm import tqdm

import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['srcnn', 'srgan'], required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    opt = parser.parse_args()

    # load model class
    if opt.model == 'srcnn':
        Model = models.SRCNNModel
    elif opt.model == 'srgan':
        Model = models.SRGANModel

    # load model state from ckpt file
    model = Model.load_from_metrics(
        weights_path=opt.ckpt,
        tags_csv=Path(opt.ckpt).parent.parent / 'meta_tags.csv',
        on_gpu=True,
        map_location=None
    )
    model.eval()
    model.freeze()

    save_dir = Path(opt.ckpt)
    save_dir = save_dir.with_name(Path(opt.ckpt).stem.replace('_ckpt_', ''))
    save_dir.mkdir(exist_ok=True)

    criterion_PSNR = models.losses.PSNR()
    criterion_SSIM = SSIM(window_size=11, reduction='mean')

    for dataset, dataloader in model.test_dataloader.items():
        psnr_mean = 0
        ssim_mean = 0

        tbar = tqdm(dataloader)
        for batch in tbar:
            img_name = batch['path'][0]
            img_lr = batch['lr']
            img_hr = batch['hr']
            img_sr = model(img_lr)

            img_hr_ = rgb_to_grayscale(img_hr)
            img_sr_ = rgb_to_grayscale(img_sr)

            psnr = criterion_PSNR(img_hr_, img_sr_).item()
            ssim = 1 - criterion_SSIM(img_hr_, img_sr_).item()
            psnr_mean += psnr
            ssim_mean += ssim

            save_image(img_sr, save_dir / f'{dataset}_{img_name}.png', nrow=1)

        psnr_mean /= len(dataloader)
        ssim_mean /= len(dataloader)
        print(f'[{dataset}] PSNR: {psnr_mean:.4}, SSIM: {ssim_mean:.4}')


if __name__ == "__main__":
    main()
