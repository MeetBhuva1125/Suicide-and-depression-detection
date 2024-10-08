
# Suicide and Depression Detection

This repository contains two notebooks developed to classify suicide and depression-related text data using machine learning models in TensorFlow. One notebook (`suicide-and-depression-detection.ipynb`) uses the DirectML plugin for hardware acceleration, while the other (`suicide-depression-detection.ipynb`) is configured to run on Kaggle with its own GPU acceleration setup.

## Notebooks

1. **suicide-and-depression-detection.ipynb**:
   - Utilizes the DirectML plugin for running models on compatible hardware.
   - Requires local dataset loading and setup.
   - Best suited for local execution on machines with DirectML-compatible GPUs.

2. **suicide-depression-detection.ipynb**:
   - Designed for Kaggle's environment, leveraging its GPU setup.
   - Data is expected to be loaded directly from Kaggle's storage.
   - Best suited for cloud-based execution.

## Prerequisites

- **Python 3.8 or above**
- **Required Packages**: These can be installed with the following command:

    ```bash
    pip install tensorflow neattext pandas scikit-learn tqdm
    ```

- **DirectML Plugin** (for local execution with DirectML hardware acceleration):

    Follow [Microsoft's DirectML plugin setup](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin) to install and configure DirectML.

## Dataset

The dataset `Suicide_Detection.csv` should be placed in the same directory as the notebooks if running locally. Kaggle users can access the dataset directly from Kaggle's storage.
[Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
## Word Embeddings

Both notebooks use **GloVe embeddings** to represent text data in a dense vector format, providing semantic context and enhancing model accuracy. Ensure you have downloaded the required GloVe embeddings (e.g., `glove.6B.100d.txt` or a similar file) and placed it in the directory or specified the path within the notebook.

## Running the Notebooks

1. **Local Execution (DirectML)**:
    - Ensure DirectML is set up on your machine.
    - Place the `Suicide_Detection.csv` file in the root directory.
    - Run the `suicide-and-depression-detection.ipynb` notebook.

2. **Kaggle Execution**:
    - Upload the notebook `suicide-depression-detection.ipynb` to your Kaggle environment.
    - Ensure the dataset is selected as an input in your Kaggle notebook settings.
    - Execute the notebook cells sequentially.

## Model Architecture

Both notebooks utilize a similar LSTM-based model architecture with the following layers:
   - Embedding Layer (with GloVe embeddings)
   - LSTM Layers (with optional Bidirectional settings)
   - Dense Layers
   - Dropout for regularization

## Results

Each notebook outputs classification reports to evaluate model performance in detecting and distinguishing between suicidal and depressive text data.

--- 

This update includes information on how to incorporate the GloVe embeddings into your project. Let me know if you'd like further adjustments!
