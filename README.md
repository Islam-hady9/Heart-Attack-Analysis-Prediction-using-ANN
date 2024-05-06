# Heart Attack Analysis Prediction using ANN

## Heart Attack Analysis & Prediction using Artificial Neural Network
![Heart Attack](https://myacare.com/uploads/AdminBlogs/91d19c6155d145348eb5dcd8b161fd36.png)

This repository contains a Jupyter notebook that explores the risk factors associated with heart attacks and builds an Artificial Neural Network (ANN) to predict heart attack occurrence based on various medical and demographic features.

## Notebook Content

### 1. **Introduction**
   - The notebook begins with an introduction to the problem statement, explaining the significance of predicting heart attacks using various clinical and demographic data.

### 2. **Data Loading**
   - Data is loaded into a pandas DataFrame. This data includes attributes like age, sex, cholesterol levels, and other relevant health metrics that could influence heart attack risk.

### 3. **Exploratory Data Analysis (EDA)**
   - **Data Inspection**: Initial checks for data types, missing values, and statistical summaries.
   - **Visualization**: Multiple visualizations are generated to understand the distribution of individual variables and their relationships with the heart attack outcome.

### 4. **Data Preprocessing**
   - **Feature Engineering**: Techniques applied include encoding categorical variables and scaling numerical variables to prepare the data for neural network modeling.
   - **Data Splitting**: The data is split into training and testing sets to evaluate the model's performance.

### 5. **Model Building**
   - **ANN Architecture**: An ANN model is constructed using TensorFlow and Keras, consisting of multiple dense layers with activation functions suited for binary classification.
   - **Compilation**: The model is compiled with an appropriate optimizer and loss function for binary classification.
   - **Training**: The model is trained on the training data using a validation split to monitor performance and prevent overfitting with an early stopping callback.

### 6. **Model Evaluation**
   - **Performance Metrics**: After training, the model is evaluated on the test set, and metrics like accuracy and loss are reported.
   - **Visualization**: Training history (loss and accuracy over epochs) is visualized to assess learning progress.

### 7. **Prediction**
   - **Making Predictions**: The trained model is used to predict heart attack risk on unseen test data.
   - **Result Analysis**: Predictions are analyzed and compared with actual labels to evaluate the model's practical utility.

### 8. **Conclusion**
   - The concluding section discusses the model's performance, insights gained from the analysis, and potential improvements for future iterations.

## Insights and Observations
- The notebook contains detailed insights derived from EDA, highlighting key risk factors and their impact on heart attack probability.
- Insights from the model's predictions are discussed, providing a practical understanding of its efficacy in real-world scenarios.

## Usage
- To run this notebook, ensure you have the required libraries installed (as listed in the dependencies section).
- Execute the cells in sequence to replicate the analysis and model training.

## Dependencies
```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn plotly
```

## Developers
- [Islam Abdelhady Hassanein](https://github.com/Islam-hady9)
- [Enas Ragab Abdel_Latif](https://github.com/EnasRagab22)
- [Mariam Tarek Saad](https://github.com/Mariam-Tarek6)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
