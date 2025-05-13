# AI/Deep Learning for Brown Fat Detection and Removal in PET/CT Scans

This project implements a two-phase approach to automatically detect the presence of brown adipose tissue (BAT) in PET/CT scans and subsequently suppress its signal in the 3D PET images to improve diagnostic clarity.

**The full project, including code and example datasets, can be downloaded from this repository: [https://github.com/abduali02/c0138/](https://github.com/abduali02/c0138/)**

## Prerequisites

Before running the scripts, ensure you have Python installed along with the following major libraries:
* PyTorch (`torch`, `torchvision`)
* `pydicom`
* `SimpleITK`
* `numpy`
* `scikit-learn`
* `TotalSegmentator`
* `imageio` (implied by DicomDataset, often a dependency)
* `scipy` (for `ndimage.label`)

You can generally install these using pip:
`pip install torch torchvision pydicom SimpleITK numpy scikit-learn TotalSegmentator imageio scipy`

Refer to `Appendix 4: Code` within the main project document (available in the repository) for specific versions if issues arise or for detailed setup.

## Phase 1: 2D CNN Classifier for Brown Fat Presence Detection

This phase uses a ResNet-50 based 2D Convolutional Neural Network (CNN) to classify PET Maximum-Intensity Projection (MIP) images for the presence of brown fat.

### How to Use Phase 1:
1.  **Download the Project**:
    * Clone or download the entire project from [https://github.com/abduali02/c0138/](https://github.com/abduali02/c0138/). This includes the necessary scripts and example data.
2.  **Locate Data**:
    * The example PET MIP images (DICOM files) are typically located within the downloaded repository (e.g., in the `dataset/dicoms/` subdirectory).
    * The corresponding CSV file mapping filenames to brown fat presence labels (e.g., `dataset/labels.csv`) is also included. A test CSV (e.g., `dataset/labels_test.csv`) is used for evaluation.
3.  **Run the Classifier Script**:
    * The script for this phase is `2D Brown fat classifier - Phase 1`.
    * Ensure paths within the script (e.g., `data_csv`, `dcm_root`, `out_dir`) correctly point to the locations within the downloaded project structure.
    * **Training**: If you wish to retrain the model using the provided data, uncomment the `train(arguments)` line. The script will save the best model (e.g., `best.pt`) to the specified output directory.
    * **Inference**: To run inference on a single DICOM file or the test set using the pre-trained model (included as `best.pt`), ensure the model path (`arguments.model`) is correctly set and use the `infer_single()` or `test_set()` functions.
4.  **Output**:
    * For inference on a single image, the script will print the predicted brown fat presence score.
    * When evaluating a test set, it provides classification metrics and a confusion matrix.

## Phase 2: 3D Brown Fat Segmentation and Removal

This phase processes the full 3D PET/CT volumes. If brown fat is deemed present, this pipeline segments and suppresses the brown fat signal in the PET image. It uses CT Hounsfield Unit (HU) filtering for adipose tissue, `TotalSegmentator` for liver segmentation (to establish a reference PET uptake), and rule-based relative intensity thresholding.

### How to Use Phase 2:
1.  **Download the Project**:
    * Clone or download the entire project from [https://github.com/abduali02/c0138/](https://github.com/abduali02/c0138/). This includes the scripts and example 3D scan data.
2.  **Locate Data**:
    * The example 3D PET and CT DICOM series are located within the downloaded repository (e.g., in subdirectories `dataset/3D/[scan_number]/PET/` and `dataset/3D/[scan_number]/CT/`). The scans provided for testing are "44", "108", "171".
3.  **Ensure TotalSegmentator is Available**:
    * `TotalSegmentator` is used for liver segmentation. Ensure it is installed and accessible in your Python environment. The script calls `totalsegmentator.python_api.totalsegmentator`.
4.  **Run the Removal Script**:
    * The script for this phase is `3D Brown fat removal - Phase 2` (found in `Appendix 4: Code` of the main project document, and available in the repository).
    * Modify the `scan_number` variable and ensure other relevant paths (e.g., `base_dicom_path`, `output_dir`, `segm_dir`) at the beginning of the script correctly point to the data within the downloaded project structure.
    * Execute the script. It will:
        * Load the CT and PET series from the specified scan number.
        * Run `TotalSegmentator` to get a liver mask from the CT (if the mask doesn't already exist).
        * Create a fat mask based on CT HU values.
        * Resample these masks to the PET image grid.
        * Calculate PET uptake in the liver to set an adaptive threshold.
        * Identify BAT voxels within the fat mask that are above this threshold.
        * Suppress the PET signal in these identified BAT voxels.
5.  **Output**:
    * The script will save the modified (brown fat suppressed) PET DICOM series to the specified `output_dir` (e.g., `dataset/3D/[scan_number]/PET_scrubbed/` within the repository structure).
    * This folder is already provided from a previous run - it should be identical assuming hyper parameters don't change.
    * A NIfTI file for the liver segmentation will also be saved in `segm_dir`.
    * Any DICOM viewer such as MicroDICOM may be used - open the whole `scan_number` directory to compare the before and after.

## Data

* The project uses DICOM files for PET and CT scans. Phase 1 uses 2D MIPs and CSV files for labels. Phase 2 uses 3D DICOM series.
* **All necessary example data for running the scripts is provided within this repository:** [https://github.com/abduali02/c0138/](https://github.com/abduali02/c0138/). Download the repository to access this data.
* The typical data structure can be found in the `dataset/` directory of the repository.

## Running the Project

1.  Clone or download the entire project from [https://github.com/abduali02/c0138/](https://github.com/abduali02/c0138/).
2.  Set up your Python environment with all the prerequisites listed above.
3.  The notebooks are included in the repository.
4.  Modify paths within the scripts if necessary to align with your local copy of the downloaded repository structure.
5.  Run the Python script for the desired phase.
6.  Alternatively, upload notebooks and data to Google Colab to run.
