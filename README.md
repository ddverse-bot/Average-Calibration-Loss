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

## Dataset: ACDC Dataset
Link: https://www.kaggle.com/datasets/anhoangvo/acdc-dataset

## Visualizations and Results
Some examples:
<img width="1773" height="597" alt="image" src="https://github.com/user-attachments/assets/b8e20f96-a0e7-4ac4-9f15-d195ea141e58" />
<img width="1777" height="597" alt="image" src="https://github.com/user-attachments/assets/fe2bcd5a-d6a7-4922-8024-2c997a006e69" />
<img width="1773" height="597" alt="image" src="https://github.com/user-attachments/assets/57dbdc62-469a-4dc9-abd9-3adfc017457c" />
<img width="1777" height="597" alt="image" src="https://github.com/user-attachments/assets/53ce3919-6940-482e-98b1-be0c7ce584bc" />
<img width="1773" height="597" alt="image" src="https://github.com/user-attachments/assets/4d09271f-3214-4966-981f-c4b0acf5ff8e" />
<img width="956" height="790" alt="image" src="https://github.com/user-attachments/assets/43f4d4c6-d9d1-4583-a479-8b2b71268376" />
<img width="956" height="790" alt="image" src="https://github.com/user-attachments/assets/f6c30373-a760-4295-bb8e-855c7ae16f8a" />

Average Dice: 0.5609
Average ECE: 0.0030
Average ACE: 0.1707
Average MCE: 0.3547

Note: Please checkout the colab notebook: https://colab.research.google.com/drive/1Js8X3f7M-xcDSvaNwVEcfeTE1FyVOWSa?usp=sharing

4.  **Visualize results:**
    ```bash
    python visualize.py
    ```
    This script displays various visualizations, including predicted masks, uncertainty maps, reliability diagrams, and 2D histograms of confidence vs. accuracy for selected samples.
