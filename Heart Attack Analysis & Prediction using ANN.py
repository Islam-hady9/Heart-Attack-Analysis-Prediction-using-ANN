# Heart Attack Analysis & Prediction using ANN

# -----------------------------------------------------

# Introduction 📖
# "Heart attacks" are a serious health issue worldwide. This analysis aims to find out which factors are connected to heart attacks and which ones affect it the most. By using data analysis and machine learning, the goal is to build a machine-learning model that can accurately predict the likelihood of someone having a heart attack. This can help people know if they are at risk and take steps to stay healthy and avoid heart attacks.

# -----------------------------------------------------

# 1. Import Libraries 📚

# EDA Libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

# Data Preprocessing Libraries
from datasist.structdata import detect_outliers
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Machine Learing and Deep Learning Libraries
import tensorflow as tf
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_regression, RFE, SelectFromModel
from imblearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, classification_report, roc_curve, roc_auc_score
from tensorflow.keras.utils import plot_model

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------

# 2. Data Exploration 🔎

df = pd.read_csv(r"Dataset\heart.csv")
df.sample(10)

# check the dataset shape
print("Number of Columns in data",df.shape[1])
print("---------------------------------------")
print("Number of Rows in data",df.shape[0])

# data information
df.info()

# checking for duplicated values
df.duplicated().sum()

# Removing duplicated data
df.drop_duplicates(inplace=True)

# checking if duplicated value has been removed
df.duplicated().sum()

# checking count the number of unique values in each column of the data
df.nunique()

# Descriptive analysis for numerical data
df.describe().style.background_gradient()

# -----------------------------------------------------

# 3. Exploratory Data Analysis 📊

# 3.1. Univariate Analysis

# Exploration: Categorical Features

fig, axes = plt.subplots(3, 3, figsize=(13, 9))

# Creating a list of categorical features 
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'output']

#Looping through the subplots and create countplots for each feature
for i, ax in enumerate(axes.flat):
    if i < len(cat_features):
        sns.countplot(data=df, x=cat_features[i], ax=ax, palette="mako", orient='h')
        ax.set_title(f'Countplot for {cat_features[i]}', fontsize=14)
        
# Adjusting the layout for better visualization
plt.tight_layout()
plt.show()

# Exploration: Numerical Features

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

# Creating a list of categorical features 
cont_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

#Looping through the subplots and create countplots for each feature
for i, ax in enumerate(axes.flat):
    if i < len(cont_features):
        sns.boxplot(data=df, x=cont_features[i], ax=ax, palette="mako", orient='h')
        ax.set_title(f'Boxplot for {cont_features[i]}', fontsize=16)
        
# Adjusting the layout for better visualization
plt.tight_layout()
plt.show()

# Skewed Continuous Features Exploration

cont_columns = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
fig, axes = plt.subplots(ncols=len(cont_columns), figsize=(18, 5))

# Plot distribution plots for each skewed column
for i, column in enumerate(cont_columns):
    sns.histplot(data=df, x=column, kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {column}', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

plt.tight_layout()
plt.show()

# 3.2. Bivariate Analysis

#The Effect of Age on Risk of Heart Attack (Output)

# Creating a histogram using Plotly Express to visualize the relationship between age and the risk of heart attack
fig = px.histogram(df, x='age', color='output', title='The Effect of Age on Risk of Heart Attack (Output)',
                   labels={'age': 'Age', 'output': 'Output'}, 
                   marginal='box', barmode='group',
                   color_discrete_sequence=['#48a890', '#234457'], text_auto=True  
                   )

# Customizing the layout of the histogram
fig.update_layout(
    xaxis=dict(tickmode='linear', dtick=2),  # Adjusting x-axis tick settings
    bargap=0.1  # Setting the gap between bars
)

# Customizing gridlines on the plot
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Customizing the background colors
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()

# The Effect of Sex on Risk of Heart Attack (Output)

# Filtering the DataFrame to separate male and female data
df_male = df[df['sex'] == 1]
df_female = df[df['sex'] == 0]

# Counting the occurrences of heart attack presence (output) for males and females
male_counts = df_male['output'].value_counts()
female_counts = df_female['output'].value_counts()

colors = ['#234457', '#48a890']  

# Creating subplots for male and female distributions
fig = make_subplots(rows=1, cols=2, subplot_titles=('Male', 'Female'), specs=[[{'type':'domain'}, {'type':'domain'}]])

# Adding a pie chart for male heart attack presence
fig.add_trace(go.Pie(values=male_counts, name='Male',
                     marker=dict(colors=colors)), 1, 1)

# Adding a pie chart for female heart attack presence
fig.add_trace(go.Pie(values=female_counts, name='Female',
                     marker=dict(colors=colors)), 1, 2)

# Customizing the hole in the pie charts
fig.update_traces(hole=.4)

# Customizing the overall layout, title, and annotations
fig.update_layout(title_text='The Effect of Sex on Risk of Heart Attack (Output)', title_font=dict(size=18), title_x=0.5, title_y=0.95,
                 annotations=[dict(text='Male', x=0.22, y=0.45, font_size=25, showarrow=False),
                 dict(text='Female', x=0.78, y=0.45, font_size=25, showarrow=False)])

# Customizing background colors
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()

# The Effect of Sex on Risk of Heart Attack (Output)

# Creating a histogram to visualize the distribution of chest pain types (cp) with respect to heart attack risk (output)
fig = px.histogram(df, x='cp', color='output', title='The Effect of Cp (Chest Pain Types) on Risk of Heart Attack (Output)',
                   labels={'cp': 'Chest Pain Types', 'output': 'Output'}, barmode='group',
                   color_discrete_sequence=['#48a890', '#234457'], text_auto=True 
                   )

# Customizing the gap between bars in the histogram
fig.update_layout(
    bargap=0.1
)

# Customizing the x-axis to show tick values and labels for different chest pain types
fig.update_xaxes(showgrid=True, gridcolor='lightgray', tickvals=[0, 1, 2, 3],
                 ticktext=['Typical Angina (0)', 'Atypical Angina (1)', 'Non-Anginal Pain (2)', 'Asymptomatic (3)'])

# Customizing the appearance of the y-axis
fig.update_yaxes(showgrid=True, gridcolor='lightgray')

# Customizing background colors
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()

# The Effect of Resting ECG Results (restecg) on Risk of Heart Attack (Output)

# Creating a histogram to visualize the effect of Resting ECG Results (restecg) on Heart Attack Risk (output)
fig = px.histogram(df, x='restecg', color='output', 
                   title='The Effect of Resting ECG Results (restecg) on Risk of Heart Attack (Output)',
                   labels={'restecg': 'Resting ECG Results (restecg)', 'output': 'Output'}, barmode='group',
                   color_discrete_sequence=['#48a890', '#234457'],
                   category_orders={'restecg': ['0', '1', '2']},  text_auto=True 
                   )

# Customizing the x-axis tick values and labels
fig.update_xaxes(tickvals=[0, 1, 2], ticktext=['Normal (0)', 'ST-T Wave Abnormality (1)', 'Probable/Definite LVH (2)'])

# Customizing the background color and gridlines
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()

# The Effect of Exercise-Induced Angina (exng) on Risk of Heart Attack (Output)

# Creating a histogram to visualize the relationship between Exercise-Induced Angina (exng) and the risk of heart attack (Output)
fig = px.histogram(df, x='exng', color='output', title='Exercise-Induced Angina (exng) vs. Risk of Heart Attack (Output)',
                   labels={'exng': 'Exercise-Induced Angina (exng)', 'output': 'Output'}, 
                   barmode='group',
                   color_discrete_sequence=['#48a890', '#234457'], text_auto=True  
                   )

# Customizing layout: adjusting the gap between bars, marker appearance, gridlines, and title
fig.update_layout(
    bargap=0.1
)
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()

# The Effect of Oldpeak on Risk of Heart Attack (Output)

# Creating a histogram to visualize the relationship between Oldpeak and the risk of heart attack (Output)
fig = px.histogram(df, x='oldpeak', color='output', title='The Effect of Oldpeak on Risk of Heart Attack (Output)',
                   labels={'oldpeak': 'Oldpeak', 'output': 'Output'}, barmode='group',
                   color_discrete_sequence=['#48a890', '#234457'], text_auto=True )

# Customizing layout: adjusting the gap between bars, marker appearance, gridlines, and title
fig.update_layout(
    bargap=0.1
)
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()

# The Effect of Slope of ST Segment (slp) on Risk of Heart Attack (Output) 

# Creating a histogram to visualize the relationship between the Slope of ST Segment (slp) and the risk of heart attack (Output)
fig = px.histogram(df, x='slp', color='output', title='The Effect of Slope of ST Segment (slp) on Risk of Heart Attack (Output)',
                   labels={'slp': 'Slope of ST Segment', 'output': 'Output'}, barmode='group',
                   color_discrete_sequence=['#48a890', '#234457'], text_auto=True )

# Customizing layout: adjusting the gap between bars, marker appearance, gridlines, and title
fig.update_layout(
    bargap=0.1
)
fig.update_xaxes(showgrid=True, gridcolor='lightgray', tickvals=[0, 1, 2, 3],
                 ticktext=['Downsloping (0)', 'Flat (1)', 'Upsloping (2)'])
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()

# The Effect of Number of Major Vessels Colored by Fluoroscopy (CAA) on Risk of Heart Attack (Output)

# Creating a histogram to visualize the relationship between the Number of Major Vessels (caa) and the risk of heart attack (Output)
fig = px.histogram(df, x='caa', color='output', barmode='group', 
                   title='The Effect of Number of Major Vessels Colored by Fluoroscopy (CAA) on Risk of Heart Attack (Output)', 
                   color_discrete_sequence=['#48a890', '#234457'],
                   labels={'caa': 'CAA (Number of Major Vessels)', 'output': 'Output'}, text_auto=True )

# Customizing layout: adjusting the title, font size, and background color
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.show()

# The Effect of Thalassemia Type (Thall) and Risk of Heart Attack (Output)

# Creating a histogram to visualize the relationship between Thalassemia Type (Thall) and the risk of heart attack (Output)
fig = px.histogram(df, x='thall', color='output', title='The Effect of Thalassemia Type (Thall) and Risk of Heart Attack (Output)',
                   labels={'thall': 'Thalassemia Type', 'output': 'Output'}, barmode='group',
                   color_discrete_sequence=['#48a890', '#234457'], text_auto=True )

# Customizing layout: adjusting the gap between bars, marker appearance, gridlines, and title
fig.update_layout(
    bargap=0.1
)
fig.update_xaxes(showgrid=True, gridcolor='lightgray', tickvals=[0, 1, 2, 3], 
                 ticktext=['None (Normal) (0)', 'Fixed Defect (1)', 'Reversible Defect (2)', 'Thalassemia (3)'])
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.show()


# 3.3. Multivariate Analysis

# The Effect of Resting Blood Pressure (trtbps) and Age on Heart Attack Risk 

# Creating a scatter plot
fig = px.scatter(df, x='age', y='trtbps', color=df['output'].astype(str),
                 title='The Effect of Resting Blood Pressure (trtbps) and Age on Heart Attack Risk ',
                 labels={'age': 'Age', 'trtbps': 'Resting Blood Pressure'},
                 color_discrete_sequence=['#48a890', '#234457'])  

# Customizing the background color and gridlines
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(legend_title_text='Output') # Rename the legend
fig.show()

# The Effect of Serum Cholesterol Levels (chol) and Age on Heart Attack Risk 

# Creating a scatter plot
fig = px.scatter(df, x='age', y='chol', color=df['output'].astype(str),
                 title='The Effect of Serum Cholesterol Levels (chol) and Age on Heart Attack Risk',
                 labels={'age': 'Age', 'chol': 'Serum Cholesterol Levels'},  
                 color_discrete_sequence=['#48a890', '#234457']) 

# Customizing the background color and gridlines
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(legend_title_text='Output') 
fig.show()

# Maximum Heart Rate During Exercise (thalachh) and Age on Heart Attack Risk

# Creating a scatter plot
fig = px.scatter(df, x='age', y='thalachh', color=df['output'].astype(str),
                 title='The Effect of Maximum Heart Rate During Exercise (thalachh) and Age on Heart Attack Risk',
                 labels={'age': 'Age', 'thalachh': 'Maximum Heart Rate During Exercise'},  
                 color_discrete_sequence=['#48a890', '#234457']) 

# Customizing the background color and gridlines
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
fig.update_xaxes(showgrid=True, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridcolor='lightgray')
fig.update_layout(legend_title_text='Output')
fig.show()

# -----------------------------------------------------

# 4. Data Preprocessing ⚒️

# 4.1. Handling Missing Data 

# checking for missing values in data
df.isna().sum()

# 4.2. Handling Categorical Data

# Working with Nominal Features with pandas `get_dummies` function.
df = pd.get_dummies(df, columns=['cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall'])

encoded = list(df.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

df.head()

# 4.3. Handling Outliers

numerical_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

# Detect outliers in numerical features
outliers_indices = detect_outliers(df, features=numerical_features, n=0)
number_of_outliers = len(outliers_indices)

print(f'Number of outliers in the Data: {number_of_outliers}')

# 4.4. Check The Distribution of Classes

# Assuming your DataFrame is named "df"
plt.figure(figsize=(6, 4))  # Adjust the figure size as needed
sns.countplot(x='output', data=df)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Class')
plt.show()

# Check the distribution of classes
print("Class distribution in dataset:")
print(df['output'].value_counts())


# 4.5. Data Split to Train and Test Sets

# First we extract the x Featues and y Label
X = df.drop(['output'], axis=1)
y = df['output']

X.shape, y.shape

# Then we Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                   )

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# 4.6. Feature Scaling

# Robust Scaling Continuous Features with RobustScaler

numerical_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

# Creating a RobustScaler instance
scaler = RobustScaler()

# Transforming (scaling) the continuous features in the training and testing data
X_train_cont_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_cont_scaled = scaler.transform(X_test[numerical_features])

# Replacing the scaled continuous features in the original data
X_train[numerical_features] = X_train_cont_scaled
X_test[numerical_features] = X_test_cont_scaled

# Display the modified X_train with scaled features
display(X_train)

# -----------------------------------------------------

# 5. ANN Model Training with Cross-Validation and Evaluation ⚙️

# 5.1. Build ANN Model Training with Cross-Validation

# The data had (boolean) values. Converting everything into (np.float32).
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# ANN Model

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


# ## This code performs K-Fold cross-validation on an artificial neural network (ANN) using TensorFlow. Let's break down each part:
# 
# ### Setting the Random Seed
# ```python
# tf.random.set_seed(42)
# ```
# This sets the random seed to 42 for TensorFlow, ensuring reproducibility of results. When you set a seed, the sequence of random numbers generated will be the same every time you run the code.
# 
# ### Defining K-Fold Cross-Validation
# ```python
# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# ```
# K-Fold cross-validation splits the dataset into `n_splits` (here, 5) folds. The `shuffle=True` parameter shuffles the data before splitting it into folds, and `random_state=42` ensures the shuffling is reproducible.
# 
# ### Preparing to Collect Scores and Histories
# ```python
# accuracies = []
# all_histories = []
# ```
# These lists will store the accuracy scores and training histories for each fold.
# 
# ### K-Fold Cross-Validation Loop
# ```python
# for train_index, val_index in kfold.split(X_train):
# ```
# This loop iterates over each fold, splitting the data into training and validation sets for each fold.
# 
# ### Splitting Data for Each Fold
# ```python
# X_train_kfold, X_val_kfold = X_train[train_index], X_train[val_index]
# y_train_kfold, y_val_kfold = y_train[train_index], y_train[val_index]
# ```
# Here, `X_train` and `y_train` are split into training and validation sets based on the indices provided by `kfold.split`.
# 
# ### Creating the Model
# ```python
# ANN_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(16, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")                                  
# ])
# ```
# A new instance of the neural network is created for each fold. This network has three layers:
# 1. An input layer with 32 neurons and ReLU activation.
# 2. A hidden layer with 16 neurons and ReLU activation.
# 3. An output layer with 1 neuron and sigmoid activation (suitable for binary classification).
# 
# ### Compiling the Model
# ```python
# ANN_model.compile(loss="binary_crossentropy",
#                   optimizer=tf.keras.optimizers.Adam(),
#                   metrics=["accuracy"])
# ```
# The model is compiled with:
# - Binary cross-entropy loss (appropriate for binary classification).
# - Adam optimizer.
# - Accuracy as the metric to evaluate performance.
# 
# ### Early Stopping Callback
# ```python
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',  
#     patience=10,  
#     restore_best_weights=True  
# )
# ```
# Early stopping monitors the validation loss and stops training if it doesn't improve for 10 epochs. It also restores the best weights to prevent overfitting.
# 
# ### Fitting the Model
# ```python
# history = ANN_model.fit(X_train_kfold, y_train_kfold, 
#                         epochs=100, 
#                         validation_data=(X_val_kfold, y_val_kfold), 
#                         callbacks=[early_stopping], 
#                         verbose=0)
# ```
# The model is trained for up to 100 epochs, using the training and validation sets for the current fold. Early stopping is applied, and the training history is recorded.
# 
# ### Collecting Histories and Evaluating the Model
# ```python
# all_histories.append(history)
# scores = ANN_model.evaluate(X_val_kfold, y_val_kfold, verbose=0)
# accuracies.append(scores[1])
# ```
# The training history is stored, and the model is evaluated on the validation set. The accuracy score is saved.
# 
# ### Printing Results
# ```python
# print("Accuracy for each fold:", accuracies)
# print("Average accuracy:", np.mean(accuracies))
# ```
# Finally, the accuracy for each fold and the average accuracy across all folds are printed.
# 
# ### Summary
# This code performs K-Fold cross-validation to evaluate the performance of an ANN on a given dataset. It ensures the model's results are reproducible, uses early stopping to prevent overfitting, and calculates the accuracy for each fold as well as the average accuracy across all folds.

ANN_model.summary()

# Visualize the model
plot_model(ANN_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# 5.2. ANN Model Evaluation

# Evaluate model on the test dataset
loss, accuracy = ANN_model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Plots training and validation accuracy and loss for each fold on shared plots, making it easy to compare performance across folds.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('Training and Validation Metrics Across Folds')

for i, history in enumerate(all_histories):
    ax1.plot(history.history['accuracy'], label=f'Fold {i+1} Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label=f'Fold {i+1} Val Accuracy', linestyle='--')
    ax2.plot(history.history['loss'], label=f'Fold {i+1} Train Loss')
    ax2.plot(history.history['val_loss'], label=f'Fold {i+1} Val Loss', linestyle='--')

ax1.set_ylabel('Accuracy')
ax1.legend()
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

# save the figure
plt.savefig('folds_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Predictions on test data
y_pred = ANN_model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
print("\nConfusion Matrix:\n", cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(cm)
disp.plot()

# save the figure
plt.savefig('cm_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Display classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))

pred = ANN_model.predict(X_test).reshape(-1)  # Reshape predictions to 1D array
pred_binary = (pred >= 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

print("True heart attack chances      :", y_test.astype(int)[:20])
print("Predicted heart attack chances :", pred_binary[:20])