# main.py
import os
import torch
import numpy as np
from glob import glob
import monai
from monai.transforms import (
    Compose, Spacingd, Orientationd,
    ScaleIntensityRanged, RandFlipd, RandRotate90d, ToTensord,
    RandCropByPosNegLabeld, SpatialPadd
)
from monai.data import CacheDataset, DataLoader

# Import custom modules
from utils.data_transforms import LoadH5d
from losses.custom_losses import dice_loss, HardL1ACELoss, SoftL1ACELoss
from models.segresnet_builder import build_segresnet

# --- 1. Setup Environment ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 2. Kaggle Setup (to be run once locally or integrated with data pipeline) ---
# For local VS Code setup, you'd typically download the dataset manually or
# ensure your Kaggle API key is set up in ~/.kaggle/kaggle.json
# For example, you might run these commands in your terminal *before* running main.py:
# mkdir -p ~/.kaggle
# cp /path/to/your/kaggle.json ~/.kaggle/
# chmod 600 ~/.kaggle/kaggle.json
# kaggle datasets download -d anhoangvo/acdc-dataset
# unzip -q acdc-dataset.zip -d acdc_dataset

data_dir = "acdc_dataset/ACDC_preprocessed/ACDC_training_volumes"
print("Dataset path:", data_dir)

# --- 3. Prepare 3D dataset list ---
if not os.path.exists(data_dir):
    # This part would only run if the data is not present, e.g., if you haven't downloaded it
    print(f"Data directory {data_dir} not found. Please ensure data is downloaded and unzipped.")
    # You might want to add code here to automate download if not already done manually
    # For simplicity, we assume data is ready.
    exit()


patients = sorted(glob(os.path.join(data_dir, "patient*.h5")))
data_dicts = []
for patient_path in patients:
    data_dicts.append({"image": patient_path, "label": patient_path})

# Split 80/20
train_split = int(0.8 * len(data_dicts))
train_files = data_dicts[:train_split]
val_files = data_dicts[train_split:]

print(f"Total patients: {len(data_dicts)}")
print(f"Training patients: {len(train_files)}")
print(f"Validation patients: {len(val_files)}")

# --- 4. MONAI Transforms ---
train_transforms = Compose([
    LoadH5d(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear","nearest")),
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
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ToTensord(keys=["image", "label"])
])

val_transforms = Compose([
    LoadH5d(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5,1.5,2.0), mode=("bilinear","nearest")),
    Orientationd(keys=["image","label"], axcodes="RAS"),
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
    ToTensord(keys=["image","label"])
])

# --- 5. Create Datasets and Loaders ---
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

print("DataLoaders ready. Train batches:", len(train_loader), "Val batches:", len(val_loader))

# --- 6. Model, Optimizer, Loss ---
model = build_segresnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    blocks_down=(1,2,2),
    blocks_up=(1,2),
    init_filters=16,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
hard_ace = HardL1ACELoss(n_bins=20)
soft_ace = SoftL1ACELoss(n_bins=20)

# --- 7. Training Loop ---
best_val_loss = float("inf")
epochs = 50
lambda_hard = 0.5
lambda_soft = 0.5

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch_data in train_loader:
        img, mask = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        pred = model(img)
        pred = torch.sigmoid(pred)
        loss = dice_loss(pred, mask) + lambda_hard*hard_ace(pred, mask) + lambda_soft*soft_ace(pred, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_data in val_loader:
            img, mask = batch_data["image"].to(device), batch_data["label"].to(device)
            pred = model(img)
            pred = torch.sigmoid(pred)
            loss = dice_loss(pred, mask) + lambda_hard*hard_ace(pred, mask) + lambda_soft*soft_ace(pred, mask)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "segresnet_acdc.pth")
        print("Saved best model: segresnet_acdc.pth")

print("Training complete!")