import os
import argparse
import numpy as np
from skimage.io import imsave
from tqdm import tqdm
import torch
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from utils.palette import colorize_mask
from models.pspnet import PSPNet
from dataloader import ATLANTIS
from torch.utils.data import DataLoader


def main(
        model,
        split,
        num_classes,
        input_size,
        padding_size,
        batch_size,
        num_workers,
        data_directory,
        restore_from,
        save_path,
):
    cudnn.enabled = True
    cudnn.benchmark = True

    if model == "PSPNet":
        model = PSPNet(img_channel=3, num_classes=num_classes)

        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

    model.eval()
    model.cuda()

    saved_state_dict = torch.load(restore_from)
    # model_dict = model.state_dict()
    # saved_state_dict = {k: v for k,
    #                     v in saved_state_dict.items() if k in model_dict}
    # model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    test_dataset = ATLANTIS(data_directory, split="test", padding_size=padding_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=False)

    interpolation = torch.nn.Upsample(size=(input_size, input_size), mode="bilinear",
                                      align_corners=True)
    with torch.no_grad():
        for image, mask, name, width, height in tqdm(test_dataloader):
            image = F.interpolate(image, size=(input_size, input_size), mode="bilinear",
                                  align_corners=True)

            # GPU deployment
            image = image.cuda()

            # Compute prediction and loss
            _, pred = model(image)

            pred = interpolation(pred).detach().cpu().numpy()[0].transpose(1, 2, 0)

            pred = np.array(np.argmax(pred, axis=2), dtype=np.uint8)
            mask = np.array(mask.squeeze(0), dtype=np.uint8)

            top_pad = padding_size - height
            right_pad = padding_size - width
            pred = pred[top_pad:, :-right_pad]

            rgb_pred = colorize_mask(pred, num_classes)
            rgb_mask = colorize_mask(mask, num_classes)
            imsave('%s/%s.png' % (args.save_path, name[0][:-4]), pred)

            if split != "test":
                rgb_pred.save('%s/%s_color.png' % (save_path, name[0][:-4]))
                rgb_mask.save('%s/%s_gt.png' % (save_path, name[0][:-4]))

        print("finish")


def get_arguments(
    MODEL="PSPNet",
    SPLIT="test",
    NUM_CLASSES=56,
    INPUT_SIZE=640,
    PADDING_SIZE=768,
    BATCH_SIZE=1,
    NUM_WORKERS=1,
    DATA_DIRECTORY="./atlantis",
    RESTORE_FROM="./snapshots/review_results/epoch28.pth",
    SAVE_PATH="./snapshots/test_review_results_epoch28"
):
    parser = argparse.ArgumentParser(description=f"Testing {MODEL} on ATLANTIS 'test' set.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Model name: {MODEL}.")
    parser.add_argument("--split", type=str, default=SPLIT,
                        help="ATLANTIS 'test' set.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict, excluding background.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Integer number determining the height and width of input image.")
    parser.add_argument("--padding-size", type=int, default=PADDING_SIZE,
                        help="Integer number determining the height and width of model output.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for multithread data loading.")
    parser.add_argument("--data-directory", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where model restores parameters from.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="Path to save results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args.model, args.split, args.num_classes, args.input_size,
         args.padding_size, args.batch_size, args.num_workers,
         args.data_directory, args.restore_from, args.save_path)
