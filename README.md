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
3. Open the `Heart_Attack_Analysis_Prediction.ipynb` file from the Jupyter Notebook dashboard.
4. Run all cells in the notebook to perform the data analysis, model training, and prediction steps.

## Model Architecture

The ANN model is constructed using the following layers:

1. Input Layer: Corresponding to the number of features in the dataset.
2. Hidden Layers: Multiple hidden layers with ReLU activation functions.
3. Output Layer: A single neuron with a sigmoid activation function to output the probability of a heart attack.

The model is trained using the Adam optimizer and binary cross-entropy loss function.

## Results

The model achieves an accuracy of XX% on the test set, with a precision of YY% and a recall of ZZ%. Detailed results and model evaluation metrics can be found in the `results` directory.

## Contributors

- Islam Abd_Elhady Hassanein (Project Lead)
- Enas Ragab Abdel_Latif
- Mariam Tarek Saad

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
