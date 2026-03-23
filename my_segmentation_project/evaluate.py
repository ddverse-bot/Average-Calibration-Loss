import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from utils.data_transforms import LoadH5d
from losses.custom_losses import dice_loss, HardL1ACELoss, SoftL1ACELoss
from models.segresnet_builder import build_segresnet

from monai.transforms import (
    Compose, Spacingd, Orientationd,
    ScaleIntensityRanged, RandFlipd, RandRotate90d, ToTensord,
    RandCropByPosNegLabeld, SpatialPadd
)
from monai.data import CacheDataset, DataLoader

def compute_calibration(confidence, error, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, ace, mce = 0.0, 0.0, 0.0
    valid_bins = 0
    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i+1])
        if np.sum(mask) == 0:
            continue
        acc = np.mean(1 - error[mask])
        avg_conf = np.mean(confidence[mask])
        diff = np.abs(avg_conf - acc)
        ece += np.sum(mask) / len(confidence) * diff
        ace += diff
        mce = max(mce, diff)
        valid_bins += 1
    ace = ace / valid_bins if valid_bins > 0 else 0
    return ece, ace, mce

def reliability_diagram(confidence, error, save_path):
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []
    for i in range(n_bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i+1])
        if np.sum(mask) == 0:
            continue
        accs.append(np.mean(1 - error[mask]))
        confs.append(np.mean(confidence[mask]))
    plt.figure()
    plt.plot(confs, accs, marker='o', label="Model")
    plt.plot([0, 1], [0, 1], '--', label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    data_dir = "acdc_dataset/ACDC_preprocessed/ACDC_training_volumes"
    patients = sorted(glob(os.path.join(data_dir, "patient*.h5")))
    data_dicts = []
    for patient_path in patients:
        data_dicts.append({"image": patient_path, "label": patient_path})

    train_split = int(0.8 * len(data_dicts))
    val_files = data_dicts[train_split:]

    val_transforms = Compose([
        LoadH5d(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128), mode="constant"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        ToTensord(keys=["image", "label"])
    ])
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    loader = val_loader # Using val_loader for evaluation

    model = build_segresnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        blocks_down=(1, 2, 2),
        blocks_up=(1, 2),
        init_filters=16
    ).to(device)
    model.load_state_dict(torch.load("segresnet_acdc.pth"))

    num_mc_dropout_samples = 5

    os.makedirs("predicted_masks", exist_ok=True)
    os.makedirs("uncertainty_maps", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    dice_scores = []
    ece_scores, ace_scores, mce_scores = [], [], []

    model.train() # Enable dropout for MC dropout inference

    for i, batch_data in enumerate(tqdm(loader, desc="Evaluating Model")):
        img, mask = batch_data["image"].to(device), batch_data["label"].to(device)

        mc_predictions = []
        with torch.inference_mode():
            for _ in range(num_mc_dropout_samples):
                pred = model(img)
                pred = torch.sigmoid(pred)
                mc_predictions.append(pred.cpu().numpy())

        mc_predictions = np.array(mc_predictions)
        mean_pred_np = mc_predictions.mean(axis=0).squeeze()
        uncertainty_map_np = mc_predictions.std(axis=0).squeeze()
        mask_np = mask.squeeze().cpu().numpy()
        pred_bin = (mean_pred_np > 0.5).astype(np.float32)

        # Save predicted mean mask
        torch.save(torch.tensor(mean_pred_np), f"predicted_masks/pred_{i:03d}.pt")
        # Save uncertainty map
        torch.save(torch.tensor(uncertainty_map_np), f"uncertainty_maps/uncertainty_{i:03d}.pt")

      
        dice_scores.append(dice_loss(torch.tensor(mean_pred_np), torch.tensor(mask_np)).item())

        
        error_map = np.abs(mask_np - pred_bin)
        confidence = 1 - np.abs(mean_pred_np - pred_bin) 

        ece, ace, mce = compute_calibration(confidence.flatten(), error_map.flatten())
        ece_scores.append(ece)
        ace_scores.append(ace)
        mce_scores.append(mce)

        
        if i < 10: 
            reliability_diagram(confidence.flatten(), error_map.flatten(), f"results/reliability_{i}.png")

    print("----- RESULTS -----")
    print(f"Average Dice: {np.mean(dice_scores):.4f}")
    print(f"Average ECE: {np.mean(ece_scores):.4f}")
    print(f"Average ACE: {np.mean(ace_scores):.4f}")
    print(f"Average MCE: {np.mean(mce_scores):.4f}")
    print("Predicted mean masks saved in 'predicted_masks/'")
    print("Uncertainty maps saved in 'uncertainty_maps/'")
    print("Reliability diagrams saved in 'results/'")
