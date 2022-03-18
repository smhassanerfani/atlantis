import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from dataloader import ATLANTIS
from torch.utils.data import DataLoader
from models.pspnet import PSPNet
import joint_transforms as joint_transforms


class AdjustLearningRate:
    num_of_iterations = 0

    def __init__(self, optimizer, base_lr, max_iter, power):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.power = power

    def __call__(self, current_iter):
        lr = self.base_lr * ((1 - float(current_iter) / self.max_iter) ** self.power)
        self.optimizer.param_groups[0]['lr'] = lr
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = lr * 10

        return lr


def train_loop(dataloader, model, loss_fn, optimizer, lr_estimator, interpolation):
    # size = len(dataloader.dataset)
    for batch, (images, masks, _, _, _) in enumerate(dataloader, 1):

        # GPU deployment
        images = images.cuda()
        masks = masks.cuda()

        # Compute prediction and loss
        aux, pred = model(images)
        aux = interpolation(aux)
        pred = interpolation(pred)
        loss = loss_fn(pred, masks) + 0.4 * loss_fn(aux, masks)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_estimator.num_of_iterations += len(images)
        lr = lr_estimator(lr_estimator.num_of_iterations)

        if batch % 100 == 0:
            loss, current = loss.item(), lr_estimator.num_of_iterations
            print(f"loss: {loss:.5f}, lr = {lr:.6f} [{current:6d}/{lr_estimator.max_iter:6d}]")


def val_loop(dataloader, model, loss_fn, interpolation):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for images, masks, _, _, _ in dataloader:

            # GPU deployment
            images = images.cuda()
            masks = masks.cuda()

            # Compute prediction and loss
            aux, pred = model(images)
            aux = interpolation(aux)
            pred = interpolation(pred)
            val_loss += loss_fn(pred, masks) + 0.4 * loss_fn(aux, masks)
            correct += (pred.argmax(1) == masks).type(torch.float).sum().item()

        val_loss /= num_batches
        correct /= (size * masks.size(1) * masks.size(2))
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")


def main(args):
    cudnn.enabled = True
    cudnn.benchmark = True

    # Loading model
    if args.model == "PSPNet":
        model = PSPNet(img_channel=3, num_classes=args.num_classes)

    try:
        os.makedirs(args.snapshot_dir)
    except FileExistsError:
        pass

    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()

    for key, value in saved_state_dict.items():
        if key.split(".")[0] not in ["head", "dsn", "fc"]:
            new_params[key] = value

    model.load_state_dict(new_params, strict=False)

    model = model.cuda()
    model.train()

    # Dataloader
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(
            args.input_size,
            False,
            pre_size=None,
            scale_min=0.5,
            scale_max=2.0,
            ignore_index=0),
        joint_transforms.Resize(args.input_size),
        joint_transforms.RandomHorizontallyFlip()]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)
    train_dataset = ATLANTIS(args.data_directory, split="train", joint_transform=train_joint_transform)
    val_dataset = ATLANTIS(args.data_directory, split="val", joint_transform=train_joint_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Initializing the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    interpolation = torch.nn.Upsample(size=(args.input_size, args.input_size), mode="bilinear",
                                      align_corners=True)

    max_iter = args.num_epochs * len(train_dataloader.dataset)
    lr_poly = AdjustLearningRate(optimizer, args.learning_rate, max_iter, args.power)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, lr_poly, interpolation)
        val_loop(val_dataloader, model, loss_fn, interpolation)
        torch.save(model.state_dict(),
                   os.path.join(args.snapshot_dir, "epoch" + str(epoch + 1) + ".pth"))
    print("Done!")


def get_arguments(
        MODEL="PSPNet",
        NUM_CLASSES=56,
        SNAPSHOT_DIR="snapshots/review_results/",
        DATA_DIRECTORY="atlantis",
        INPUT_SIZE=640,
        BATCH_SIZE=2,
        NUM_WORKERS=4,
        LEARNING_RATE=2.5e-4,
        MOMENTUM=0.9,
        WEIGHT_DECAY=0.0001,
        NUM_EPOCHS=30,
        POWER=0.9,
        RESTORE_FROM="snapshots/resnet101-imagenet.pth"
):
    parser = argparse.ArgumentParser(description=f"Training {MODEL} on ATLANTIS.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Model Name: {MODEL}")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict, excluding background.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to restore the model parameters.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of s")
    parser.add_argument("--data-directory", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for multithreading dataloader.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epochs for training.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args)
