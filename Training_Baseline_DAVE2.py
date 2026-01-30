import time
#import random
import argparse
import contextlib
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda import amp
from Network_Baseline import Dave2Regression
from Dataset_Baseline import get_dataloader


def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # faster with fixed shapes
        use_amp = True
        scaler = amp.GradScaler()
        pin = True
    else:
        use_amp = False
        scaler = None
        pin = False

    model = Dave2Regression(out_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    nr_epochs  = 100
    batch_size = 16
    start_time = time.time()

    train_loader = get_dataloader(data_folder, batch_size, num_workers=4,shuffle=True,pin_memory=True)
    best_loss = float('inf')
    history = {
        "loss": [],
        "mae_thr": [],
        "mae_str": [],
        "mae_brk": [],
        "rmse_thr": [],
        "rmse_str": [],
        "rmse_brk": [],
    }


    for epoch in range(nr_epochs):
        model.train()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        for imgs, actions_row in train_loader:
            imgs     = imgs.to(device, non_blocking=pin)
            actions_row = actions_row.to(device, non_blocking=pin)

            optimizer.zero_grad(set_to_none=True)
            ctx = amp.autocast() if use_amp else contextlib.nullcontext()
            with ctx:
                pred = model(imgs)
                huber_loss = regression_loss(pred, actions_row)
                gas_brake_pen = (pred[:, 1] * pred[:, 2]).mean()
                loss = huber_loss + 0.1 * gas_brake_pen                         #we want to penalize if it slams both gas and brake


            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            all_preds.append(pred.detach().cpu())
            all_targets.append(actions_row.detach().cpu())


        epoch_loss = total_loss / max(1, num_batches)
        scheduler.step()
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = time_per_epoch * (nr_epochs - 1 - epoch)
        with torch.no_grad():
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            mae  = torch.mean(torch.abs(all_preds - all_targets), dim=0)              # (3,)
            rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2, dim=0))      # (3,)
        print(
            f"Epoch {epoch+1:3d}  "
            f"loss: {epoch_loss:.4f}  "
            f"MAE[s/t/b]: {mae[0]:.3f}/{mae[1]:.3f}/{mae[2]:.3f}  "
            f"RMSE[s/t/b]: {rmse[0]:.3f}/{rmse[1]:.3f}/{rmse[2]:.3f}  "
            f"ETA +{time_left:.1f}s"
        )
        history["loss"].append(epoch_loss)
        history["mae_str"].append(mae[0].item())
        history["mae_thr"].append(mae[1].item())
        history["mae_brk"].append(mae[2].item())
        history["rmse_str"].append(rmse[0].item())
        history["rmse_thr"].append(rmse[1].item())
        history["rmse_brk"].append(rmse[2].item())


        torch.save(model.state_dict(), save_path)
        if epoch_loss < best_loss:
          best_loss = epoch_loss
          torch.save(model.state_dict(), save_path.replace(".pth", "_best.pth"))
    epochs = range(1, nr_epochs + 1)

    plt.figure(figsize=(10,6))
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_regression.png")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(epochs, history["mae_thr"], label="Throttle MAE")
    plt.plot(epochs, history["mae_str"], label="Steering MAE")
    plt.plot(epochs, history["mae_brk"], label="Brake MAE")
    plt.title("Mean Absolute Error per Control Channel")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_mae_per_channel.png")
    plt.show()

def regression_loss(pred, targets):
    """
    Calculates the regression loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    # Return mean loss across the batch
    return F.smooth_l1_loss(pred, targets, reduction='mean')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC518 Homework1 Imitation Learning')
    parser.add_argument('-d','--data_folder', default="./", type=str,help='path to data folder (images/, sensors.npy, aligned_index.npy, optional sensor_norm.json)')
    parser.add_argument('-s', '--save_path', default="./model_latest.pth", type=str, help='where to save the .pth model')
    args = parser.parse_args()

    train(args.data_folder, args.save_path)