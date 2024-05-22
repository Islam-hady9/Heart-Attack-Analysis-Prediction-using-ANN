# Heart Attack Analysis Prediction using ANN

<div align="center">
  <p align="center">
    <img src="https://myacare.com/uploads/AdminBlogs/91d19c6155d145348eb5dcd8b161fd36.png" alt="Heart Attack" />
  </p>
<p align="center">
<strong>Heart Attack Analysis & Prediction using Deep Learning (Artificial Neural Network)</strong></p>
</div>

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Contributors](#contributors)
8. [License](#license)

## Introduction

This project involves the analysis and prediction of heart attacks using Artificial Neural Networks (ANN). The goal is to develop a predictive model that can accurately identify the likelihood of a heart attack based on various health metrics and patient data.

## Dataset

The dataset used for this project is sourced from the [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/data). It includes features such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, the slope of the peak exercise ST segment, number of major vessels, and thalassemia.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Islam-hady9/Heart-Attack-Analysis-Prediction-using-ANN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Heart-Attack-Analysis-Prediction-using-ANN
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the entire Jupyter Notebook for analysis and prediction, follow these steps:

1. Ensure the dataset is in the correct format and available in the project directory.
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the `Heart Attack Analysis & Prediction using ANN.ipynb` file from the Jupyter Notebook dashboard.
4. Run all cells in the notebook to perform the data analysis, model training, and prediction steps.

## Model Architecture

The ANN model is constructed using the following layers:

1. Input Layer: Corresponding to the number of features in the dataset.
2. Hidden Layers: Multiple hidden layers with ReLU activation functions.
3. Output Layer: A single neuron with a sigmoid activation function to output the probability of a heart attack.

The model is trained using the Adam optimizer and binary cross-entropy loss function.

Here is a simplified code snippet of the model architecture:
```python
# Set the random seed for reproducibility
tf.random.set_seed(42)
# Define the number of folds for KFold cross-validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# Prepare to collect scores and histories
accuracies = []
all_histories = []

# KFold Cross Validation
for train_index, val_index in kfold.split(X_train):
    # Split data
    X_train_kfold, X_val_kfold = X_train[train_index], X_train[val_index]
    y_train_kfold, y_val_kfold = y_train[train_index], y_train[val_index]
    
    # Create a new instance of the model (to reinitialize weights)
    ANN_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")                                  
    ])
    
    # Compile the model
    ANN_model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  
        patience=10,  
        restore_best_weights=True  
    )
    
    # Fit the model
    history = ANN_model.fit(X_train_kfold, y_train_kfold, 
                        epochs=100, 
                        validation_data=(X_val_kfold, y_val_kfold), 
                        callbacks=[early_stopping], 
                        verbose=0)  # Set verbose to 0 to reduce output
    
    # Collect the history from each fold
    all_histories.append(history)
    
    # Evaluate the model on the validation set
    scores = ANN_model.evaluate(X_val_kfold, y_val_kfold, verbose=0)
    accuracies.append(scores[1])  # Assume that the accuracy is the second metric
# Print the accuracy for each fold
print("Accuracy for each fold:", accuracies)
# Print the average accuracy
print("Average accuracy:", np.mean(accuracies))
```

### Model Architecture Diagram
![Model Architecture](https://github.com/Islam-hady9/Heart-Attack-Analysis-Prediction-using-ANN/blob/main/Plots_Outputs/model_plot.png)

## Results

The model achieves an accuracy of 88.5% on the test set, with a precision of 87% and a recall of 87%.

### Confusion Matrix Visualization
![Confusion Matrix Plot](https://github.com/Islam-hady9/Heart-Attack-Analysis-Prediction-using-ANN/blob/main/Plots_Outputs/cm_plot.png)

### Plots Training and Validation Accuracy and Loss for Each Fold on Shared Plots:
![Folds Plot](https://github.com/Islam-hady9/Heart-Attack-Analysis-Prediction-using-ANN/blob/main/Plots_Outputs/folds_plot.png)

## Contributors

- Islam Abd_Elhady Hassanein (Project Lead)
- Enas Ragab Abdel_Latif
- Mariam Tarek Saad

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
