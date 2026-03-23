import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random


from utils.data_transforms import LoadH5d
from monai.transforms import (
    Compose, Spacingd, Orientationd,
    ScaleIntensityRanged, RandFlipd, RandRotate90d, ToTensord,
    RandCropByPosNegLabeld, SpatialPadd
)
from monai.data import CacheDataset, DataLoader
from glob import glob


def setup_val_ds():
    
    data_dir = "acdc_dataset/ACDC_preprocessed/ACDC_training_volumes"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please ensure data is downloaded and unzipped.")
        return None

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
    return CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)


if __name__ == "__main__":
   
    print("--- Visualizing Predicted Masks, Uncertainty Maps, and Reliability Diagrams ---")
    all_available_indices = list(range(10)) 
    random_indices_for_display = random.sample(all_available_indices, min(10, len(all_available_indices)))

    print(f"Displaying predicted masks, uncertainty maps, and reliability diagrams for samples: {random_indices_for_display}")

    for i, sample_idx in enumerate(random_indices_for_display):
        pred_filename = f'predicted_masks/pred_{sample_idx:03d}.pt'
        uncertainty_filename = f'uncertainty_maps/uncertainty_{sample_idx:03d}.pt'
        reliability_filename = f'results/reliability_{sample_idx}.png'

        if not os.path.exists(pred_filename) or not os.path.exists(uncertainty_filename) or not os.path.exists(reliability_filename):
            print(f"Skipping sample {sample_idx}: Required files not found. This might happen if not enough samples were evaluated or files were moved.")
            continue

        predicted_mask = torch.load(pred_filename).cpu().numpy()
        uncertainty_map = torch.load(uncertainty_filename).cpu().numpy()

        middle_slice_idx = uncertainty_map.shape[0] // 2

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(predicted_mask[middle_slice_idx, :, :], cmap='gray')
        plt.title(f'Sample {sample_idx}: Predicted Mask (Central Slice)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(uncertainty_map[middle_slice_idx, :, :], cmap='hot')
        plt.title(f'Sample {sample_idx}: Uncertainty Map (Central Slice)')
        plt.colorbar(label='Uncertainty (Std Dev)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        reliability_img = plt.imread(reliability_filename)
        plt.imshow(reliability_img)
        plt.title(f'Sample {sample_idx}: Reliability Diagram')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    print("Finished displaying selected samples with reliability diagrams.")

    
    print("\n--- Visualizing 2D Histograms of Confidence vs. Accuracy ---")
    val_ds = setup_val_ds() 
    if val_ds is None:
        print("Could not set up validation dataset for 2D histogram visualization.")
        exit()

    all_val_sample_indices = list(range(len(val_ds)))
    indices_to_show_histograms = random.sample(all_val_sample_indices, min(3, len(all_val_sample_indices)))

    print(f"Displaying 2D histograms for samples: {indices_to_show_histograms}")

    for sample_idx in indices_to_show_histograms:
        pred_filename = f'predicted_masks/pred_{sample_idx:03d}.pt'
        if not os.path.exists(pred_filename):
            print(f"Skipping sample {sample_idx}: Predicted mask file not found: {pred_filename}")
            continue

        mean_pred_np = torch.load(pred_filename).cpu().numpy()
        sample_data = val_ds[sample_idx]
        mask_np = sample_data[0]["label"].cpu().numpy().squeeze()

        pred_bin = (mean_pred_np > 0.5).astype(np.float32)
        error_map = np.abs(mask_np - pred_bin)
        confidence = 1 - np.abs(mean_pred_np - pred_bin)

        confidence_flat = confidence.flatten()
        accuracy_flat = (1 - error_map).flatten()

        plt.figure(figsize=(10, 8))
        plt.hist2d(confidence_flat, accuracy_flat, bins=(50, 2), cmap='Blues')
        plt.colorbar(label='Density of Pixels')
        plt.xlabel('Confidence Bins')
        plt.ylabel('Accuracy')
        plt.title(f'2D Histogram of Confidence vs. Accuracy for Sample {sample_idx}')
        plt.yticks([0.25, 0.75], ['Incorrect (0)', 'Correct (1)'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    print("Finished displaying additional 2D histograms.")
