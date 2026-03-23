# ACDC Segmentation with SegResNet and Calibration Losses

This project implements 3D cardiac image segmentation using MONAI's SegResNet model on the ACDC dataset. It incorporates custom calibration-aware loss functions (Hard L1 ACE and Soft L1 ACE) to not only improve segmentation accuracy but also ensure the model's predictions are well-calibrated, meaning its confidence scores accurately reflect its likelihood of being correct.

## Project Structure

```
my_segmentation_project/
├── data/                 # Raw and preprocessed dataset files
├── models/               # Model definitions or builders
│   └── segresnet_builder.py
├── losses/               # Custom loss functions
│   └── custom_losses.py
├── utils/                # Utility functions, e.g., custom data transforms
│   └── data_transforms.py
├── main.py               # Main script for data setup, training, and model saving
├── evaluate.py           # Script for model evaluation, MC Dropout, and calibration metrics
├── visualize.py          # Script for generating and displaying visualizations
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and instructions
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd my_segmentation_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Kaggle API Setup & Data Download:**
    *   Ensure you have a Kaggle account and have generated an API key (`kaggle.json`).
    *   Place `kaggle.json` in `~/.kaggle/` (e.g., `/home/youruser/.kaggle/kaggle.json` on Linux/macOS, or `C:\Users\YourUser\.kaggle\kaggle.json` on Windows).
    *   Set appropriate permissions for the Kaggle key:
        ```bash
        chmod 600 ~/.kaggle/kaggle.json
        ```
    *   Download and unzip the ACDC dataset. You can do this manually or via command line:
        ```bash
        kaggle datasets download -d anhoangvo/acdc-dataset
        unzip -q acdc-dataset.zip -d data/acdc_dataset
        # You might need to adjust the data_dir in main.py, evaluate.py, visualize.py
        # to point to data/acdc_dataset/ACDC_preprocessed/ACDC_training_volumes
        ```

## Usage

1.  **Train the model:**
    ```bash
    python main.py
    ```
    This will train the SegResNet model and save the best checkpoint as `segresnet_acdc.pth`.

2.  **Evaluate the model:**
    ```bash
    python evaluate.py
    ```
    This script performs Monte Carlo Dropout inference, calculates Dice and calibration metrics, and saves predicted masks, uncertainty maps, and reliability diagrams to `predicted_masks/`, `uncertainty_maps/`, and `results/` folders respectively.

3.  **Visualize results:**
    ```bash
    python visualize.py
    ```
    This script displays various visualizations, including predicted masks, uncertainty maps, reliability diagrams, and 2D histograms of confidence vs. accuracy for selected samples.