# 🐾 Animal Classification using CNN

 A multi-class image classification project using a **Convolutional Neural Network (CNN)** built from scratch with **TensorFlow/Keras**.

 The model classifies animal images into **5 different categories**:

 - 🐱 `cat`
- 🐄 `cow`
- 🦌 `deep` (deer)
- 🐶 `dog`
- 🦁 `lion`

 The project uses the **Animal Image Classification, 5 Species** dataset from Kaggle. The dataset is already divided into training, validation, and testing sets.

---

 ## 📌 Project Overview

 The objective of this project is to build a basic CNN capable of recognizing different animal species from images.

 The complete pipeline includes:

 1. Downloading the dataset
2. Exploring the dataset folder structure
3. Loading and preprocessing images
4. Applying data augmentation
5. Building a CNN from scratch
6. Training and validating the model
7. Evaluating model performance
8. Generating a confusion matrix and classification report
9. Predicting individual images
10. Saving the trained model

---

 ## 📂 Dataset

 **Dataset:** Animal Image Classification, 5 Species

 **Source:** Kaggle

 **Classes:**

```
cat
cow
deep
dog
lion
```

 > Note: The dataset uses `deep` as the folder/class name for deer.

 The dataset contains approximately **629 images** across the five classes and is already split into:

```
dataset/
├── train/
│   ├── cat/
│   ├── cow/
│   ├── deep/
│   ├── dog/
│   └── lion/
│
├── validation/
│   ├── cat/
│   ├── cow/
│   ├── deep/
│   ├── dog/
│   └── lion/
│
└── test/
    ├── cat/
    ├── cow/
    ├── deep/
    ├── dog/
    └── lion/
```

---

 ## 🛠️ Technologies Used

 - **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **KaggleHub**
- **Google Colab** (recommended for running the notebook)

---

 ## 📦 Installation

 Install the required KaggleHub package:

```
pip install -q kagglehub
```

 The other libraries are generally available in Google Colab. If running locally, install the dependencies with:

```
pip install tensorflow numpy matplotlib seaborn scikit-learn kagglehub
```

---

 ## 🚀 How to Run

 ### 1\. Clone the Repository

```
git clone https://github.com/your-username/animal-classification-cnn.git
cd animal-classification-cnn
```

 ### 2\. Open the Notebook

 Open the provided Jupyter/Colab notebook.

 If using Google Colab, upload the notebook and run the cells sequentially.

 ### 3\. Download the Dataset

 The notebook automatically downloads the dataset using KaggleHub:

```
import kagglehub

dataset_path = kagglehub.dataset_download(
    "miadul/animal-image-classification-5-species"
)

print("Path to dataset files:", dataset_path)
```

 If Kaggle authentication is required, log in with your Kaggle account.

---

 ## 🖼️ Image Preprocessing

 Images are resized to:

```
128 × 128 pixels
```

 The images are loaded using Keras:

```
image_dataset_from_directory()
```

 Pixel values are normalized from:

```
[0, 255]
```

 to:

```
[0, 1]
```

 using:

```
Rescaling(1./255)
```

---

 ## 🔄 Data Augmentation

 Since the dataset is relatively small, light data augmentation is applied to the training images.

 The following transformations are used:

 - Random horizontal flipping
- Random rotation
- Random zoom

```
data_augmentation = Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

 Data augmentation helps reduce overfitting and improves the model's ability to generalize to unseen images.

---

 ## 🧠 CNN Architecture

 The model is a simple CNN built from scratch using TensorFlow/Keras.

 ### Architecture

```
Input: 128 × 128 × 3
        ↓
Rescaling (1/255)
        ↓
Conv2D (32 filters, 3×3)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filters, 3×3)
        ↓
MaxPooling2D
        ↓
Conv2D (128 filters, 3×3)
        ↓
MaxPooling2D
        ↓
Dropout (0.25)
        ↓
Flatten
        ↓
Dense (128 neurons)
        ↓
Dropout (0.50)
        ↓
Dense (5 neurons)
        ↓
Softmax
```

 The final **Softmax** layer produces a probability for each of the five animal classes.

---

 ## ⚙️ Model Configuration

 The model is compiled using:

```
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

 ### Training Settings

 | Parameter | Value |
| --- | --- |
| Image Size | 128 × 128 |
| Batch Size | 16 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Output Activation | Softmax |
| Number of Classes | 5 |

---

 ## ⏹️ Early Stopping

 Because the dataset is small, **EarlyStopping** is used to reduce overfitting.

```
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

 Training stops when the validation loss no longer improves for several consecutive epochs, and the best model weights are restored.

---

 ## 📊 Model Evaluation

 The model is evaluated using the test dataset.

 The project generates:

 - Test accuracy
- Test loss
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrix
- Classification report
- Individual image predictions

 Example:

```
test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

---

 ## 📈 Training Curves

 The notebook plots training and validation performance to visualize whether the model is learning effectively.

 ### Accuracy

 The accuracy graph compares:

```
Training Accuracy
Validation Accuracy
```

 ### Loss

 The loss graph compares:

```
Training Loss
Validation Loss
```

 These plots can be used to identify potential overfitting or underfitting.

---

 ## 🔲 Confusion Matrix

 A confusion matrix is generated using `scikit-learn` and visualized with Seaborn.

```
cm = confusion_matrix(y_true, y_pred)
```

 The matrix shows how many images from each actual class were correctly or incorrectly classified.

 Example structure:

```
              Predicted
           cat cow deep dog lion

Actual cat
Actual cow
Actual deep
Actual dog
Actual lion
```

---

 ## 📋 Classification Report

 The project also generates a classification report containing:

 - Precision
- Recall
- F1-score
- Support

```
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
```

 This provides a more detailed evaluation of performance for each animal class.

---

 ## 🔮 Predicting an Individual Image

 A helper function is included to classify an individual image.

```
predict_image(str(TEST_DIR / "lion" / "some_image.jpg"))
```

 The function displays:

 - Input image
- Predicted animal class
- Prediction confidence

 Example output:

```
Prediction: lion
Confidence: 94.25%
```

---

 ## 💾 Saving the Model

 The trained model is saved in Keras format:

```
model.save("animal_classification_cnn.keras")
```

 The saved model can later be loaded using:

```
model = tf.keras.models.load_model(
    "animal_classification_cnn.keras"
)
```

---

 ## 📁 Suggested Repository Structure

```
animal-classification-cnn/
│
├── README.md
├── SMIT Animal_Classification_CNN_Assignment - MFQ.ipynb
├── animal_classification_cnn.keras
│
├── results/
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
└── .gitignore
```

 > The dataset itself does not need to be uploaded to GitHub. It can be downloaded through KaggleHub when running the notebook.

---

 ## 🎯 Results

 After training, the notebook reports the actual test performance:

```
Test Loss: <your test loss>
Test Accuracy: <your test accuracy>
```

 The exact results may vary slightly depending on the TensorFlow/Keras version, hardware, and training environment.

 For a more complete evaluation, refer to the generated:

 - Accuracy/Loss curves
- Confusion matrix
- Classification report

---

 ## 🌟 Key Learning Outcomes

 This project demonstrates how to:

 - Work with an image classification dataset
- Load images using TensorFlow/Keras
- Perform image preprocessing
- Apply data augmentation
- Build a CNN from scratch
- Use Softmax for multi-class classification
- Train a deep learning model
- Apply early stopping
- Evaluate classification performance
- Generate confusion matrices
- Generate precision/recall/F1 reports
- Make predictions on individual images
- Save and reload a trained Keras model

---

 ## ⚠️ Limitations

 This project uses a relatively small dataset of approximately **629 images**.

 Therefore, the model may not generalize well to real-world animal images.

 Potential limitations include:

 - Small number of training images
- Limited variation in backgrounds
- Different lighting conditions may affect predictions
- Some animal classes may have fewer examples
- Similar-looking animals may be confused
- The model is designed primarily for educational purposes

 For real-world deployment, a much larger and more diverse dataset would be recommended.

---

 ## 🔮 Future Improvements

 Possible improvements include:

 - Increasing the dataset size
- Using stronger data augmentation
- Adding Batch Normalization
- Using a deeper CNN architecture
- Hyperparameter tuning
- Using Transfer Learning with models such as MobileNetV2, ResNet, or EfficientNet
- Applying learning-rate scheduling
- Performing cross-validation
- Deploying the model as a web application
- Creating a Streamlit interface for image prediction

---

 ## 📚 Dataset Reference

 This project uses the **Animal Image Classification, 5 Species** dataset available on Kaggle.

 Dataset:

 `miadul/animal-image-classification-5-species`

---

 ## 👨‍💻 Project Type

 **Machine Learning / Deep Learning**

 **Task:** Multi-Class Image Classification

 **Model:** Convolutional Neural Network (CNN)

 **Framework:** TensorFlow / Keras

 **Number of Classes:** 5

---

 ## 📜 License

 This repository is intended for **educational and academic purposes**.

 Please refer to the original Kaggle dataset for its specific license and usage terms.

---

 ## ⭐ Acknowledgements

 Thanks to the dataset creator for providing the Animal Image Classification, 5 Species dataset on Kaggle.

 If you found this project useful, consider giving the repository a ⭐.
