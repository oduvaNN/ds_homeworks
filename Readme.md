# Data Science Homework Assignment 1: Exploratory Data Analysis and Dimensionality Reduction

## Overview

This assignment covers exploratory data analysis (EDA) and dimensionality reduction (DR) techniques. It involves:

- **EDA:** Analyzing individual and pairwise features to understand distributions, anomalies, and correlations.
- **Data Preprocessing:** Handling missing values, outliers, and preparing data for DR.
- **DR:** Implementing techniques like PCA and t-SNE, visualizing results, and selecting optimal components.

# Data Science Homework Assignment 2: Regression Analysis of Air Quality Dataset

## Overview

This assignment focuses on regression analysis of the Air Quality dataset, with the target variable being C6H6(GT). The task involves:

- **EDA:** Conducting univariate and multivariate analyses to explore the data distribution and relationships.
- **Data Preparation:** Handling missing values, normalizing/scaling data, and performing basic feature engineering.
- **Baseline Model:** Implementing a linear regression model without regularization.
- **Evaluation:** Choosing appropriate evaluation metrics, performing residual analysis, and drawing conclusions.
- **Hyperparameters Tuning:** Optimizing model hyperparameters for improved performance.
- **Feature Importance:** Assessing the importance of features in the regression model.

# Data Science Homework Assignment 3: Classification Task on WeatherAUS Dataset

## Overview

This assignment focuses on a classification task using the WeatherAUS dataset, with the target variable being 'RainTomorrow'. The task involves:

- **EDA:** Conducting univariate and multivariate analyses to explore the dataset, with conclusions drawn from the analysis.
- **Data Preparation:** Handling missing values, normalization, and encoding, with comments provided for each step.
- **Metrics Selection:** Choosing appropriate evaluation metrics and providing reasoning for each metric selected.
- **Modeling:** Implementing various classification algorithms including Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machine (SVM). This includes training, evaluation, and hyperparameter tuning for each model.
- **Interpretation and Analysis:** Interpreting coefficients for logistic regression, plotting feature importance, and identifying optimal thresholds. Demonstrating the impact of different K values for KNN. Analyzing the ROC curve and calculating AUC for each model.
- **Handling Imbalanced Data:** Implementing techniques such as under-sampling, oversampling, weighting, and stratification to address class imbalance and commenting on the results.
- **Comprehensive Model Evaluation:** Comparing all models with each other and drawing conclusions. Comparing the original models with modified versions and analyzing improvements.

# Data Science Homework Assignment 4: Tree Models for Regression and Classification Tasks

## Overview

This assignment focuses on utilizing tree models for regression and classification tasks using datasets from previous modules. The tasks included:

- **Regression Task:** Predicting C6H6(GT) levels using the Air Quality dataset.
- **Classification Task:** Predicting 'RainTomorrow' using the WeatherAUS dataset.

### Highlights

- **Regression Problem:** Implemented Decision Tree Regressor, Random Forest Regressor, and Boosting Regressor (XGBoost/LightGBM). Conducted thorough evaluation, hyperparameter tuning, and feature importance analysis.
- **Classification Problem:** Utilized Decision Tree Classifier, Random Forest Classifier, and Boosting Classifier (XGBoost/LightGBM) with similar evaluation and tuning processes.
- **Model Comparison:** Evaluated new models against those from previous assignments to determine the best-performing model for both regression and classification tasks.

# Data Science Homework Assignment 5: Clustering Analysis on Earthquake Dataset

## Overview

This assignment focuses on conducting clustering analysis on the earthquake dataset to identify patterns and group earthquake occurrences based on various features.

### Highlights

- **Exploratory Data Analysis and Data Preparation:** Conducted thorough exploratory data analysis and prepared the data for clustering, providing insightful comments along the way.
- **k-Means Algorithm:** Implemented the k-Means algorithm with 15 clusters and determined the optimal number of clusters using multiple methods.
- **Visualized the obtained clusters and compared k-Means with its mini-batch implementation, drawing conclusions.**
- **Other Clustering Algorithms:** Explored alternative clustering algorithms, fine-tuning hyperparameters, and providing explanations for algorithm selection and parameter choices.
- **Metrics Evaluation:** Evaluated the quality of clusters using both internal and external metrics, utilizing k-Means cluster labels as ground truth for external metrics.
- **Final Choice and Interpretation:** Explained the rationale behind the selection of the best clustering algorithm and provided interpretations of the resulting clusters.
- **Visualization on World Map:** Visualized the clustering results on a world map to provide geographical insights.

# Data Science Homework Assignment 6: Deep Learning and Convolution Operations

## Overview

This assignment involves solving a classification task using both PyTorch and PyTorch Lightning frameworks, as well as implementing convolution operations in 1D.

- **Network Construction:** Built network with appropriate loss function, layers, and other components.
- **Data Pipeline:** Implemented data pipeline including feature cleaning, handling missing values, and converting to torch tensors.
- **Network Training:** Successfully trained the network and explained findings, hyperparameters, and architectures tried.
- **PyTorch Lightning Implementation:** Built PyTorch Lightning training scripts and trained the network correctly.
- **Custom Convolution Function:** Implemented a convolution operation function without using packages like Torch or TensorFlow, explaining the tradeoff between kernel size.
- **Custom Convolution Function for Images:** Implemented a convolution operation function for 1D arrays representing images, with an additional grade coupled with showing how the custom operation works on real images.
- **Sobel Operator and Grayscale Image:** Demonstrated the convolution operator with Sobel operator on a grayscale image and drew conclusions.
