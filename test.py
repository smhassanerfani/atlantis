import os
import argparse
import numpy as np
from skimage.io import imsave
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from utils.palette import colorize_mask
from models.pspnet import PSPNet
from dataloader import ATLANTIS
from torch.utils.data import DataLoader


def main(args):
    cudnn.enabled = True
    cudnn.benchmark = True

    if args.model == "PSPNet":
        model = PSPNet(img_channel=3, num_classes=args.num_classes)

    model.eval()
    model.cuda()

    try:
        os.makedirs(args.save_path)
    except FileExistsError:
        pass

    saved_state_dict = torch.load(args.restore_from)
    # model_dict = model.state_dict()
    # saved_state_dict = {k: v for k,
    #                     v in saved_state_dict.items() if k in model_dict}
    # model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    test_dataset = ATLANTIS(args.data_directory, split="test", padding_size=args.padding_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

    interpolation = torch.nn.Upsample(size=(args.padding_size, args.padding_size), mode="bilinear",
                                      align_corners=True)
    with torch.no_grad():
        for image, mask, name, width, height in tqdm(test_dataloader):

            # GPU deployment
            image = image.cuda()

            # Compute prediction and loss
            _, pred = model(image)

            pred = interpolation(pred).detach().cpu().numpy()[0].transpose(1, 2, 0)

            pred = np.array(np.argmax(pred, axis=2), dtype=np.uint8)
            mask = np.array(mask.squeeze(0), dtype=np.uint8)

            top_pad = args.padding_size - height
            right_pad = args.padding_size - width
            pred = pred[top_pad:, :-right_pad]

            rgb_pred = colorize_mask(pred, args.num_classes)
            rgb_mask = colorize_mask(mask, args.num_classes)
            imsave('%s/%s.png' % (args.save_path, name[0][:-4]), pred)

            if args.split != "test":
                rgb_pred.save('%s/%s_color.png' % (args.save_path, name[0][:-4]))
                rgb_mask.save('%s/%s_gt.png' % (args.save_path, name[0][:-4]))

        print("finish")


def get_arguments(
    model="PSPNet",
    split="test",
    num_classes=56,
    padding_size=768,
    batch_size=1,
    num_workers=1,
    data_directory="./atlantis",
    restore_from="./snapshots/review_results/epoch30.pth",
    save_path="./snapshots/test_review_results_epoch30"
):
    parser = argparse.ArgumentParser(description=f"Testing {model} on ATLANTIS 'test' set.")
    parser.add_argument("--model", type=str, default=model,
                        help=f"Model name: {model}.")
    parser.add_argument("--split", type=str, default=split,
                        help="ATLANTIS 'test' set.")
    parser.add_argument("--num-classes", type=int, default=num_classes,
                        help="Number of classes to predict, excluding background.")
    parser.add_argument("--padding-size", type=int, default=padding_size,
                        help="Integer number determining the height and width of model output.")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=num_workers,
                        help="Number of workers for multithread data loading.")
    parser.add_argument("--data-directory", type=str, default=data_directory,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--restore-from", type=str, default=restore_from,
                        help="Where model restores parameters from.")
    parser.add_argument("--save-path", type=str, default=save_path,
                        help="Path to save results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args)
