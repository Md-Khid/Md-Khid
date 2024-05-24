# Project Overview

## Introduction
For this project, we will analyse a credit facility dataset, which can be accessed from https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download. Employing Python coding within Jupyter Notebook, we will construct a machine learning model called Artificial Neural Network (ANN). The main goal of this analysis is to predict the 'loan_status' column assessing the likelihood of a current customer defaulting on their debt payment. 

## Dataset Information

### Data Variables
The dataset consists of different details about customers' financial situations which include their personal information and factors related to loans. These details cover things like demographics, financial status and signs of how reliable they are with loans known as loan grade. These aspects form the basis of our analysis and the process of predicting outcomes . [Dataset](https://github.com/Md-Khid/ANN_Classification_Prediction/blob/main/credit_risk_dataset.csv)


### Data Dictionary
| Variable                    | Description                                     |
|-----------------------------|-------------------------------------------------|
| person_age                  | Age                                             |
| person_income               | Annual Income                                   |
| person_home_ownership       | Home Ownership: (Mortgage, Other Own, Rent)     |
| person_emp_length           | Employment Length (in years)                    |
| loan_intent                 | Loan Intent: (Debt Consolidation, Education, Home Improvement, Medical, Personal, Venture) |
| loan_grade                  | Loan Grade: (A, B, C, D, E, F, G)               |
| loan_amnt                   | Loan Amount                                     |
| loan_int_rate               | Interest Rate                                   |
| loan_status                 | Loan Status: (0 = Non Default, 1 = Default)     |
| loan_percent_income        | Percent Income                                  |
| cb_person_default_on_file  | Credit Bureau-History Default: (No, Yes)        |
| cb_person_cred_hist_length | Credit Bureau-Credit History Length             |


## Data Preparation

In this phase of data processing, we will refine the dataset for analysis by addressing missing values, handling special characters and encoding variables. Additionally, we will import all necessary modules and libraries for the project and transform categorical variables into category columns for data visualisation purposes.

### Data Pre-processing:

#### Import Python Libraries and Modules for Data Preprocessing, Modelling and Evaluation
```
# Install pip packages if required
#!pip install imblearn
#!pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Set a standard seaborn colour palette
sns.set_palette("colorblind")
```

#### Loading and Categorising Loan Status in a Credit Risk Dataset

```
# Load and read the CSV file
df = pd.read_csv('credit_risk_dataset.csv')

# Map loan_status to categorical values
loan_status_mapping = {0: 'Non-Default', 1: 'Default'}
df['loan_status'] = df['loan_status'].map(loan_status_mapping)

# Display the updated DataFrame
df
```
![1](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/74e0c63b-fae4-4830-90a2-a2bc5deea269)


#### Check Data Types 
```
# Check data types of each column
df.dtypes
```

![2](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/5e5fba1f-df18-45e6-bdb7-b3b8ff54a158)

#### Create Descriptive Stats table
```
# Create Descriptive Stats table 
Descriptive_Stats = df.describe(include='all').round(2)

# Separate columns into categorical and numerical groups
categorical_columns = Descriptive_Stats.select_dtypes(include=['object']).columns
numeric_columns = Descriptive_Stats.select_dtypes(exclude=['object']).columns

# Order columns (categorical followed by numerical)
ordered_columns = list(categorical_columns) + list(numeric_columns)
Descriptive_Stats = Descriptive_Stats.reindex(ordered_columns, axis=1)

# Transpose Descriptive Stats table 
Descriptive_Stats = Descriptive_Stats.transpose()

Descriptive_Stats
```


![3](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/84593df1-73e6-4694-a293-1657cc54e8bc)

By creating a descriptive statistics table, we can summarise important details about the dataset including central measures and spread for both categories and numbers. From the table, we notice that most customers are renting and have a good credit rating of A for their loans which are mainly for education. With an average customer age of 27 years, it seems likely that many in the dataset are working adults pursuing further studies. Moreover, we observe that the predictor column (loan_status) has imbalanced data where roughly 78% of entries fall under the Non-Default category. Interestingly, there is an anomalous Max value of 123 years contained in the "person_emp_length" column which will be addressed by removing it as an outlier.


#### Identify Columns with Missing Values
```
# Calculate number of missing values 
missing_values = df.isnull().sum()

# Filter the missing_values
columns_with_missing_values = missing_values[missing_values > 0]

# Display columns with missing values
columns_with_missing_values
```

![4](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/75816545-5fbd-4f55-9dcf-208cd45574a4)

Based on the output, it seems that the columns "person_emp_length" and "loan_int_rate" contain some missing values. To address this issue, we can decide on the most appropriate method for replacing the missing values. Possible approaches include using the mean, median or mode depending on the data distribution.

#### Fill missing values
```
# Fill missing values in 'person_emp_length' and 'loan_int_rate' columns with median
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

# Display number of columns with missing values
count_missing_values = df.isnull().sum().sum()
count_missing_values
```

![5](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/2e57a7e7-2e1b-423c-9598-6767b8ba10a7)

#### Define special characters
```
# Define special characters
special_chars = "!@#$%^&"

# Iterate over each column 
for column in df.columns:
    # Iterate over each row in current column
    for index, value in df[column].items():
        # Check if value contains any special characters
        if any(char in special_chars for char in str(value)):
            print(f"Special characters found in column '{column}', row {index}: {value}")
```
We will proceed with computing missing values for every column in the dataset and fill them with the suitable statistical value such as the median. Additionally, we will examine for any special characters that might impede the machine learning algorithm process.

## Exploratory Data Analysis 
In this section, we will delve into comprehending the dataset. This encompasses tasks such as examining data distributions, identifying outliers, visualising correlations between variables and detecting any irregularities or trends, then transforming the insights obtained into valuable information.

#### Scaling Numerical Features
```
# Define the list of numeric column names
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df
```

![6](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/d078cfbc-b1e3-4799-8d6c-52a80eef9c9e)

We will scale the numerical value columns in the dataset. This helps us understand how the variables relate to each other especially when making scatterplots and studying correlations. This ensures that the variables are standardised to a consistent scale, thereby facilitating accurate interpretation of their relationships.

#### Correlation Matrix Plot
```
def plot_corr_and_print_highly_correlated(df):
    # Calculate correlation matrix for numeric columns
    corr_matrix = df.select_dtypes(include='number').corr()

    # Plot heatmap
    plt.figure(figsize=(10, 5))  # Specify the figure size
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, cbar_kws={'orientation': 'vertical'})
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.show()

    # Print highly correlated pairs
    print("Highly Correlated Features:")
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8: 
                pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                print(pair)

# Call the function
plot_corr_and_print_highly_correlated(df)
```
![7](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/daa6459c-5d18-41f3-b478-44a261598ea7)

Upon examining the correlation plot, it is apparent that there is not significant correlation among the column variables except for the 'cb_person_cred_hist_length' and 'person_age' columns.

#### Scatterplot
```
# Define the list of variables for hue
hue_variables = ['loan_status','person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(35, 15))

# Flatten axes for easier iteration
axes = axes.flatten()

# Iterate over hue variables and create subplots
for i, hue_var in enumerate(hue_variables):
    sns.scatterplot(data=df, x='loan_amnt', y='loan_int_rate', hue=hue_var, ax=axes[i])
    axes[i].set_xlabel('Loan Amount')
    axes[i].set_ylabel('Loan Percent Income')
    
    # Get handles and labels for the legend
    handles, labels = axes[i].get_legend_handles_labels()
    
    # Create a custom legend with a title
    axes[i].legend(handles=handles, labels=labels, title=hue_var, loc='center left', bbox_to_anchor=(1, 0.5))

# Hide empty subplots
for i in range(len(hue_variables), axes.size):
    fig.delaxes(axes.flatten()[i])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
```
![8](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/7d5eec17-ed2f-4efb-9acc-275733890a94)

However, by colouring the plots with the categorical data columns, we can observe some interesting insights about the customers in the dataset. From the plots, we can observe the following points:


Based on the Loan Status plot: The plot suggests a higher proportion of individuals classified as non-defaulters compared to defaulters. Moreover, there is an observed trend suggesting that default rates tend to rise as loan amounts increase. Notably, loans characterised by smaller amounts and lower loan percent income exhibit a higher likelihood of being repaid resulting in a non-default status.

Based on the Home Ownership plot: The plot suggests that individuals who rent or hold mortgages often apply for larger loan amounts. Interestingly, most of the borrowers are from the mortgagors and renters group.

Based on the Loan Intent plot: It seems there is no apparent correlation between the purpose of the loan and either the loan amount or loan percent income. However, a notable concentration of data points is observed on the left-hand side of the plot. This concentration may suggest that the majority of borrowers, regardless of their loan intent, tend to seek loans with moderate amounts and are offered moderate loan amounts relative to their income.

Based on the Loan Grade plot: The plot shows that loans assigned higher grades (A and B) correspond to lower loan percent income, while loans with lower grades (E and F) entail higher loan percent income. This could suggest that higher grades (like A and B) tend to have borrowers who owe less relative to their income. On the other hand, loans with lower grades (E and F) have borrowers who owe more relative to their income, indicating higher financial risk.

Based on the Default on File plot: The plot suggests that individuals without a history of default tend to obtain loans across a broader range of amounts and generally at lower interest rates compared to those with a default record.


#### Bar Chart
```
# Select only the categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Define subplot layout
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12)) 

# Plot count for each unique item within each categorical column
for i, ax in enumerate(axes.flatten()):
    if i < len(categorical_columns):
        column = categorical_columns[i]
        order = df[column].value_counts().index  # Arrange the bars from highest to lowest count
        sns.countplot(x=column, data=df, order=order, ax=ax)
        ax.set_title('')
        ax.set_ylabel('')  # Remove the ylabel
        ax.tick_params(axis='x', rotation=45)  # Rotate x-labels by 45 degrees
    else:
        fig.delaxes(ax)  # Hide empty subplot

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
```
![9](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/b964f51c-d149-4c0c-b21c-41d227419e6d)

Based on the Home Ownership chart: The data indicates a higher number of renters compared to homeowners or mortgage holders. This trend may stem from various factors. For example, younger individuals or those in transitional life stages may prefer renting over owning a home. Economic aspects like housing affordability and credit accessibility could also influence this choice.

Based on the Loan Intent chart: The majority of loans are designated for 'Education'. This trend may imply that many individuals, particularly younger ones pursuing further education, seek financial aid to manage the escalating costs of education.

Based on the Loan Grade chart: Most loans fall into grades 'A' and 'B'. This suggests that a considerable number of individuals maintain commendable creditworthiness. This likely reflects their financial prudence and effective debt management skills.

Based on the Loan Status chart: Analysis of 'Default' and 'Non-Default' statuses reveals that the majority of loans are 'Non-Default'. This indicates successful loan repayment management by most individuals. However, the presence of 'Default' loans underscores the financial challenges faced by some individuals.

Based on the Credit Bureau Default on File chart: A significant proportion of individuals have no defaults recorded with the credit bureau. This serves as another positive indicator of their creditworthiness and may suggest a history of responsible borrowing practices

#### Dummy Variables
```
# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Remove loan_status from categorical_columns
categorical_columns.remove('loan_status')

# Perform one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_columns)

# Map loan_status to {Non-Default=0, Default=1}
df['loan_status'] = df['loan_status'].map({'Non-Default': 0, 'Default': 1})

# Display the resulting DataFrame
df
```
![10](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/5c9e32f6-5986-4f49-ad63-812dd251cdeb)

By converting categorical variables into dummy variables and assigning the loan_status as (Non-Default=0, Default=1), we can prepare them for input into machine learning models. This ensures that categorical data seamlessly integrates into the modeling process, enhancing analysis and prediction tasks performed by the algorithms.

#### Remove Outliers
```
# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Detect outliers using z-score
z_scores = stats.zscore(df[numeric_cols])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)

# Remove outliers from DataFrame
df = df[filtered_entries]

# Check the shape of the DataFrame after removing outliers
df.shape
```
![11](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/0e696da7-0fc0-4f07-8272-7d1faf8f4e70)

We will be excluding any outliers found in the dataset. This is because outliers can greatly affect how the model parameters are estimated especially for the ANN loss function during data training. By removing these outliers, we can create a more precise and reliable ANN model.

#### Class Imbalance Correction Using SMOTE
```
X = df.drop(columns=['loan_status'])  
y = df['loan_status']  

# Initialize SMOTE
smote = SMOTE()

# Perform SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save X_resampled and y_resampled to CSV
X_resampled.to_csv('X_resampled.csv', index=False)
y_resampled.to_csv('y_resampled.csv', index=False)

# Before SMOTE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Before SMOTE')
counts = df['loan_status'].value_counts().sort_index()
counts.plot(kind='pie', autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(counts) / 100))
plt.ylabel('')

# After SMOTE
plt.subplot(1, 2, 2)
plt.title('After SMOTE')
resampled_counts = y_resampled.value_counts().sort_index()
resampled_counts.plot(kind='pie', autopct=lambda p: '{:.1f}% ({:,.0f})'.format(p, p * sum(resampled_counts) / 100))
plt.ylabel('')

plt.show()
```
![12](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/58f66107-ed1b-42b7-ae00-048d89b348d7)

To address the imbalance in the predictor column (loan_status) as mentioned earlier, we will use SMOTE to oversample the minority class (Default=1) to align with the majority class (Non-Default=0). Upon applying SMOTE, we can see that both classes now have equal balanced representation.

### ANN Model Training for Loan Status Prediction
```
# Separate features and target variable
X = X_resampled  
y = y_resampled 
    
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the ANN model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
```
For the modelling phase, we will split the dataset into training and testing datasets. We will use 70% of the data to train and 30% to test. To ensure reproducible results, we will set the random state to 42. We will intend to create an Artificial Neural Network (ANN) model comprising two concealed layers. In the initial layer, we will incorporate 64 neurons utilising the Rectified Linear Unit (ReLU) as their activation function. Following this, the second layer will consist of 32 neurons, also employing ReLU as their activation function. Finally, the output layer will feature a single neuron employing a sigmoid activation function specifically designed for binary classification output.
To further configure the ANN model, we will set the following hyperparameters: ‘Stochastic Gradient Descent’ (SGD) as the optimiser for adjusting the weights of the neural network during training, Binary Crossentropy' as the loss function to help the ANN model make better predictions for binary classification tasks and ‘Accuracy’ as the measure to monitor and assess the model's precision in its classification process. Additionally, we will use the fit function to train the model. We will train the model using the X_resampled and Y_resampled datasets and feed the entire dataset to undergo 30 epochs with a batch size of 32 samples to enhance training efficiency. Furthermore, we will allocate additional validation set of 20% to help the model detect overfitting.

#### Evaluation of Model - Confusion Matrix

```
# Predictions on test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary predictions

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# Plot confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Add legend box for metrics
metrics_legend = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall (Sensitivity): {recall:.2f}\nF1 Score: {f1:.2f}\nSpecificity: {specificity:.2f}"
plt.text(1.40, 0.5, metrics_legend, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

# Add total instances for each class below the legend box
total_0_instances = np.sum(conf_matrix[0])
total_1_instances = np.sum(conf_matrix[1])
plt.text(1.30, 0.3, f'Total Instances of 0: {total_0_instances}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.text(1.30, 0.2, f'Total Instances of 1: {total_1_instances}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

```
![13](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/48dd6ece-3c98-4200-b778-a2085bae0e40)

To assess how well the ANN model performs, we employ various matrices to gauge its predictive ability. We will create a confusion matrix and analyse its performance metrics in predicting the loan_status outcome. Based on the matrix:

- Accuracy: Shows that 82% of all cases were accurately predicted by the model.
- Precision: Indicates that 87% of the predicted Default cases were accurately identified as true positives.
- Recall (Sensitivity): Shows that 76% of the actual Default cases were identified by the model.
- F1 Score: The model achieved a reasonable balance between precision and recall at 0.81. This suggests that it can effectively identify relevant instances (high recall) while also minimising false positives (high precision).
- Specificity: Indicates that 89% of Non-Default cases were correctly predicted by the model.

#### Evaluation of Model - Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)
```
# Calculate ROC curve and AUC for training set
fpr_train, tpr_train, _ = roc_curve(y_train, model.predict(X_train))
auc_train = auc(fpr_train, tpr_train)

# Calculate ROC curve and AUC for testing set
fpr_test, tpr_test, _ = roc_curve(y_test, model.predict(X_test))
auc_test = auc(fpr_test, tpr_test)

# Plot both ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (AUC = {:.2f})'.format(auc_train))
plt.plot(fpr_test, tpr_test, color='orange', lw=2, label='Test ROC curve (AUC = {:.2f})'.format(auc_test))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```

![14](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/4a226ba9-7ca0-42a8-ab92-0e34d7da1653)

We can delve deeper into assessing and demonstrating the effectiveness of the ANN model by creating a ROC and AUC chart. Based on the chart:

-ROC Curve: Both the training and testing ROC curves are notably above the diagonal red dashed line which signifies a no-skill classifier. This indicates that the model demonstrates strong predictive performance.

-AUC Value: The Area Under the Curve (AUC) for both the training and testing data is 0.90. This indicates that the model demonstrates a high degree of separability (where 0.50 denotes random chance) and can effectively differentiate between positive and negative classes.

## Evaluate Model Performance

#### Save Trained Model
```
# Save the trained model
model.save('ANN.model.h5')
```
#### Feed New Data Into Trained Model
```
# Load the new data from CSV
test = pd.read_csv('Test.data.csv')

# Define the list of numeric column names
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()
test[numeric_cols] = scaler.fit_transform(test[numeric_cols])

# Identify categorical columns
categorical_columns = test.select_dtypes(include=['object']).columns.tolist()

# Perform one-hot encoding for categorical variables
test = pd.get_dummies(test, columns=categorical_columns)

# Load the model and perform prediction
model = load_model('ANN.model.h5')

# Use X_test features for prediction
X_test = test.drop('loan_status', axis=1) 
predictions = model.predict(X_test)

# Convert probabilities into class labels
threshold = 0.5
class_predictions = [0 if pred < threshold else 1 for pred in predictions]

# Compare actual and predicted loan_status values
actual_loan_status = test['loan_status'].values
predicted_loan_status = class_predictions

# Create a DataFrame to compare actual and predicted loan_status values
comparison_df = pd.DataFrame({'Actual_loan_status': actual_loan_status, 'Predicted_loan_status': predicted_loan_status})

# Print the comparison DataFrame
comparison_df
```

![15](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/d6518c2e-e977-456b-b5dd-6bf8c3b347ea)

Using the [test](https://github.com/Md-Khid/ANN_Classification_Prediction/blob/main/Test.Data.csv) dataset, the trained ANN model accurately predicted about 85% for instances likely to be defaulters and non-defaulters. This indicates that the trained model generalised well to new data.

Predictor of Importance
```
# Load the saved model
model = keras.models.load_model('ANN.model.h5')

# Get the column names from the DataFrame
column_names = X_resampled.columns

# Retrieve the weights of the first layer of the neural network model
weights = model.layers[0].get_weights()[0]

# Calculate the absolute sum of weights for each feature
feature_weights_sum = abs(weights).sum(axis=0)

# Create a dictionary of feature names and their importance
feature_importance = dict(zip(column_names, feature_weights_sum))

# Sort the dictionary by value in descending order
sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

# Select the top 5 features
top_5_features = dict(list(sorted_feature_importance.items())[:5])

# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(list(top_5_features.keys()), list(top_5_features.values()))
plt.xlabel('Importance')
plt.ylabel('Predictors')
plt.title('Top 5 Features Importance')
plt.gca().invert_yaxis() 
plt.show()
```


![16](https://github.com/Md-Khid/ANN_Classification_Prediction/assets/160820522/95e4dffa-17bf-44e8-9f3a-049622d6cc17)

As the ANN model is often seen as a 'black box,' it can be difficult to figure out which factors truly affect the model's classification predictions. This is because of the complex multiple layers and parameters that are being used in the model’s algorithm. Therefore, it is important to determine the predictors that have the most impact on the model's predictions in order to gain insights into which inputs influence the model's decision-making process. This is particularly helpful for interpreting and identifying the key predictors that influence the model's predictions. These predictors can then be used to explain the model's behaviour to stakeholders or domain experts. Based to the features:

1.	person_home_ownership_RENT: This emerges as the most critical predictor. It suggests that the home ownership status particularly renting, has a significant influence on the model’s predictions. This could be due to various factors such as financial instability associated with renting. Thereby, increasing the risk of the event the model is predicting.
2.	loan_intent_PERSONAL: This feature ranks as the second most important. It indicates that the intent behind the loan, especially when it’s for personal use, strongly correlates with the outcome predicted by the model. This might be attributed to the fact that personal loans can be used for a variety of purposes, some of which might carry a higher risk.
3.	cb_person_cred_hist_length: This stands as the third most influential predictor. The length of a person’s credit history serves as a reliable indicator of their financial behaviour and reliability. Thereby, impacting the model’s predictions.
4.	person_age: This predictor is the fourth most significant. It suggests that the age of the person plays a substantial role in the model’s predictions. This could be due to various factors such as older individuals having more stable financial situations or younger individuals being more likely to take risks.
5.	loan_intent_EDUCATION: This feature ranking fifth in importance indicates that the intent behind the loan, particularly when it’s for educational purposes, significantly influences the model’s predictions. This may be attributed to factors such as the high cost of education which could increase the likelihood of the event the model is predicting.


### Concluding Remarks
Artificial Neural Networks (ANNs) offer a promising avenue for transforming credit risk assessment within the banking sector. However, it's crucial to recognise the vital role of human insight when analysing data. Even with advanced algorithms like ANNs, certain aspects of risk analysis rely on human judgment and intuition. Take for instance, the essential task of handling missing values and detecting outliers during data preprocessing. This critical step requires careful consideration and input from banking experts. Mishandling of missing values and outliers could distort results and potentially obscure valuable insights.

Additionally, improving the model's effectiveness involves integrating economic indicators into the datasets. This task demands a deep comprehension of financial markets, a knowledge that seasoned experts possess. While ANNs excel at processing large volumes of data, they often lack the contextual understanding needed to interpret variables accurately within the dynamic landscape of economic indicators. Therefore, the input and insights of human analysts are crucial in ensuring the meaningful interpretation and prediction of the model.


