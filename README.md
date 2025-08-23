# Suicide and Depression Detection from Text

This project aims to detect text messages that may indicate suicidal ideation. A deep learning model using a Long Short-Term Memory (LSTM) network is trained on a dataset of posts from Reddit's r/SuicideWatch subreddit. The goal is to classify a given text as either "suicide" or "non-suicide".

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [How to Use](#-how-to-use)
- [Requirements](#-requirements)
## 🚀 Project Overview

The project leverages Natural Language Processing (NLP) and Deep Learning techniques to build a binary text classifier. The model is trained to distinguish between texts that are potentially related to suicide and those that are not. This can be a valuable tool for social media monitoring and mental health support systems.

## 💾 Dataset

The model is trained on the "Suicide Detection" dataset available on Kaggle.

- **Source:** [Kaggle: Suicide Watch/Suicide Detection](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- **Description:** The dataset contains a collection of text posts. Each post is labeled with a class:
  - `suicide`: The text indicates suicidal thoughts.
  - `non-suicide`: The text is neutral.
- **Size:** The dataset consists of over 232,074 text samples, balanced between the two classes.

## 🛠️ Methodology

The project follows a standard pipeline for building a text classification model:

1.  **Data Cleaning and Preprocessing:**
    - Text is converted to lowercase.
    - Special characters and stopwords are removed using the `neattext` library.

2.  **Tokenization and Padding:**
    - The cleaned text is tokenized using `tf.keras.preprocessing.text.Tokenizer`.
    - All sequences are padded to a fixed length of 100 to ensure uniform input size.

3.  **Word Embeddings:**
    - Pre-trained GloVe embeddings (`glove.twitter.27B.200d`) are used to convert words into dense 200-dimensional vectors.
    - An embedding matrix is created to map words from our vocabulary to their corresponding GloVe vectors. This matrix is used to initialize the weights of the Keras `Embedding` layer.

4.  **Model Architecture:**
    The model is a Sequential Keras model with the following layers:
    - **Input Layer:** Expects padded sequences of length 100.
    - **Embedding Layer:** Initializes word vectors using the pre-trained GloVe embedding matrix (200 dimensions). It is set to be non-trainable.
    - **LSTM:** A Long Short-Term Memory layer with 128 units to capture sequential information.
    - **GlobalMaxPooling1D:** Reduces the dimensionality of the LSTM output.
    - **Dense & Dropout Layers:** A series of fully connected layers (256, 128, and 64 units) with ReLU activation for learning higher-level features, with Dropout for regularization.
    - **Output Layer:** A single Dense neuron with a sigmoid activation function to output a probability score for the "suicide" class.

5.  **Training:**
    - **Optimizer:** Adam with a learning rate of 0.001.
    - **Loss Function:** `binary_crossentropy`, suitable for binary classification tasks.
    - **Callbacks:**
        - `EarlyStopping`: To halt training when the validation loss stops improving.
        - `ReduceLROnPlateau`: To decrease the learning rate if training stagnates.

## 📊 Results

The model's performance is evaluated using a classification report, which includes precision, recall, and F1-score for each class.
**Test Data Classification Report (Example):**
```
               precision    recall  f1-score   support

non-suicide       0.94      0.93      0.93     23336
    suicide       0.93      0.94      0.93     23079

   accuracy                           0.93     46415
  macro avg       0.93      0.93      0.93     46415
weighted avg      0.93      0.93      0.93     46415
```
*(Note: These are representative results from a previous model version. Actual results may vary based on training.)*

## 💡 How to Use

The trained model (`model_27B_200d.h5`) and tokenizer (`tokenizer.pkl`) are saved. You can use them to make predictions on new text.

```python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and model
# Update paths if they are saved in a different directory
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = load_model('model_27B_200d.h5')

# Example text
twt = ["I'm so tired of everything, I don't see the point anymore."]

# Preprocess the text
twt_seq = tokenizer.texts_to_sequences(twt)
twt_pad = pad_sequences(twt_seq, maxlen=100)

# Make a prediction
prediction = model.predict(twt_pad)[0][0]

print(f"Prediction Score: {prediction}")

if prediction > 0.5:
    print("Result: Potential Suicide Post")
else:
    print("Result: Non Suicide Post")

```

## ⚙️ Requirements
The project requires the following Python libraries:
- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`
- `neattext`
- `tqdm`
- `matplotlib`
