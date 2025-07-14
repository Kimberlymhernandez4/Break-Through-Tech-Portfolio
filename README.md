# Break-Through-Tech-Portfolio
<img width="1024" height="512" alt="image" src="https://github.com/user-attachments/assets/dc25a42d-cc2e-4667-a7c0-1a8d5b2ae2ef" />

- My Data Science, Machine Learning, and Artificial Intelligence Portfolio developed through Cornell University's Break Through Tech Fellowship program (2025-2026). 

-----------------------------------------------------------------------------------------------------------------------------
# Unit 1: Machine Learning in a Nutshell (Introduction)
**Main Goal:** Understand the machine learning life cycle and explore common machine learning packages. 

-----------------------------------------------------------------------------------------------------------------------------
# Key Concepts Covered:
## **What is Machine Learning?** 
- It is a broad class of methods and algorithms used to build predictive models from data without directing a specific relationship between inputs and outputs.
- While ML is a subset of Artificial Intelligence in the data science field, several patterns are trained into ML models to make predictions on future data. 

-----------------------------------------------------------------------------------------------------------------------------
## **Features and Labels:** 
**What is a Feature?**
- **Features** are measureable properties that can be used to make predictions for the label (target variable) of an ML model. Some examples include age, income, or temperature depending on the dataset's focus and/or context. 
  
**What is a Label?**
- A **label** is the target output the model aims to predict, such as a customer’s purchase decision (yes/no) or the price of a house.

-----------------------------------------------------------------------------------------------------------------------------
## Supervised and Unsupervised Learning: 
**Supervised Learning:** 
- A class of ML training that uses labeled data to associate data values for classification.
- Supervised learning uses features and labels to determine its predictions within a dataset.
- Examples include linear and logistic regression. 

**Unsupervised Learning:**
- A class of ML training that uses unlabeled data to identify hidden patterns, structures, or groupings within the dataset without predefined labels or outcomes.
- Examples include clustering and dimensionality reduction techniques, such as K-Means and Principal Component Analysis (PCA).

-----------------------------------------------------------------------------------------------------------------------------
## The CRISP-DM Process as a Machine Learning Engineer
<img width="1400" height="750" alt="image" src="https://github.com/user-attachments/assets/bd923f7d-7141-4295-b86c-4ee37e7ea2ec" />

The CRISP-DM (Cross-Industry Standard Process for Data Mining) is a widely used process that guides the development of Data Science and Machine Learning projects. 
- The CRISP-DM process ensures a systematic and repeatable approach to problem-solving across different domains.

**The process consists of six major phases:**

**1. Business Understanding**
- Define project objectives, success criteria, and business goals. This phase ensures the machine learning work is aligned with real-world outcomes.

**2. Data Understanding**
- Collect initial data and explore it to identify data quality issues, patterns, or relationships that may guide modeling. Techniques include descriptive statistics and data visualization.

**3. Data Preparation**
- Transform raw data into a format suitable for modeling. This includes cleaning, feature selection, encoding categorical variables, and normalizing or scaling values.

**4. Modeling**
- Apply machine learning algorithms to the prepared data. Select appropriate models, fine-tune hyperparameters, and validate performance using techniques like cross-validation.

**5. Evaluation**
- Assess how well the model meets business goals and interpret the results. Compare different models using metrics like accuracy, precision, recall, RMSE, or R².

**6. Deployment**
- Implement the model into a production environment for real-time or batch predictions. This phase may also include monitoring and retraining plans to ensure long-term performance.

**Key Takeaway:** The CRISP-DM framework promotes iteration between phases, allowing data scientists to refine their understanding and improve models continuously.

-----------------------------------------------------------------------------------------------------------------------------
## Vectorization with NumPy
<img width="1000" height="328" alt="image" src="https://github.com/user-attachments/assets/d80fd45c-8937-45a6-8b14-a29ce15f4098" />

- **Vectorization** allows mathematical functions to be applied directly to entire arrays or matrices without using explicit loops (i.e., for and while loops), significantly improving coding speed and efficiency.
- Using **NumPy's vectorized operations** **(e.g., np.dot, np.mean, np.sum)** enables faster data processing and is essential for building scalable machine learning pipelines.


-----------------------------------------------------------------------------------------------------------------------------
## Generalization and Heuristics in ML
- **Generalization** is a model's ability to adapt to new, previously unseen data. 
- **Heuristics** are a way of solving problems where the main goal is to produce a solution within a reasonable time frame or range of accuracy. 

-----------------------------------------------------------------------------------------------------------------------------

# Unit 2: Manage Data in Machine Learning
- Build and clean a data set suitable for ML applications, add labels and features appropriate for supervised learning, and use exploratory analysis to understand the data. 

-----------------------------------------------------------------------------------------------------------------------------

# Key Concepts Covered:

## **Feature Engineering in ML**
<img width="1200" height="630" alt="image" src="https://github.com/user-attachments/assets/20f4719b-c6c1-4c47-9e18-e03dbf5eeb1e" />

- Feature engineering is the process of creating, selecting, or transforming variables in a dataset to improve the performance of machine learning models.
- It includes techniques such as normalization, binning, interaction terms, encoding categorical variables, and creating domain-specific features.
- Effective feature engineering helps models learn patterns more accurately and generalize better to unseen data.
-----------------------------------------------------------------------------------------------------------------------------

## **What is One-Hot Encoding, and Why is it Important in Machine Learning?**
<img width="524" height="295" alt="image" src="https://github.com/user-attachments/assets/7e7c1bec-24d3-4719-940b-50908d6f91e4" />

- One-Hot Encoding is a technique used to convert categorical variables into a binary matrix where each unique category becomes its own feature column.
- It is important because most machine learning algorithms require numerical input. One-Hot Encoding prevents models from interpreting categorical values as ordinal or hierarchical.


-----------------------------------------------------------------------------------------------------------------------------
## **An In-Depth Guide to Labels**
- Labels are the target variable that a machine learning model is trained to predict. In supervised learning, each data instance is associated with a label.

**Labels can be:**
- Categorical, for classification problems (e.g., spam vs. not spam),
- Numerical, for regression problems (e.g., predicting housing prices).

**Key Takeaway:** Ensuring that labels are clean, correctly matched with features, and well-distributed across classes is critical for training reliable models.

-----------------------------------------------------------------------------------------------------------------------------
## **Winsorization**
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/ba0c077f-55de-4718-967b-33ad3543691b" />

- The transformational process of statistics that is done by setting extreme outliers equal to a specified percentile of the data.
- The purpose of **Winsorization** is to reduce the effect of having too many outliers, which can reduce data quality, accuracy, and precision.
- **Applicable Example:** If the 95th percentile of income is $100,000, all values above that may be set to $100,000 to prevent skewed values in model training.

-----------------------------------------------------------------------------------------------------------------------------
## **Scatter Plots and their Value in Classification Problems**
<img width="432" height="288" alt="image" src="https://github.com/user-attachments/assets/084394b6-7ced-47fd-9694-262a4b474038" />

- Scatter plots are useful for visualizing the distribution and relationship between two features, especially when color-coded by class labels.
- In classification problems, they help detect linear or non-linear separability, clusters, and potential decision boundaries.
- They are often used in exploratory data analysis to detect feature combinations that could lead to better classification performance.

-----------------------------------------------------------------------------------------------------------------------------

# Unit 3: Train Common ML Models
- Train with two popular supervised learning algorithms: K-Nearest Neighbors (KNN) and Decision Trees (DT).
- Learn the structure of non-parametric models in supervised learning and focus on the steps needed to train a Machine Learning model. 


-----------------------------------------------------------------------------------------------------------------------------

# Key Concepts Covered:

## **An Introduction to Model Training**
- Model training is the process of feeding a machine learning algorithm with data so it can learn the relationships between input features and output labels.
- During training, the model adjusts its internal parameters or structure to minimize a loss function and improve its ability to make accurate predictions.

-----------------------------------------------------------------------------------------------------------------------------
## **What are Non-Parametric Models?**
- Non-parametric models do not assume a fixed number of parameters. Instead, they adapt their complexity based on the size and structure of the training data.
- These models can capture complex patterns and are particularly useful when the data distribution is unknown. Examples include KNN and Decision Trees.

-----------------------------------------------------------------------------------------------------------------------------
## **K-Nearest Neighbors**
- KNN is a simple, instance-based learning algorithm that classifies new data points based on the majority class of their K closest neighbors in the training set.
- **Note:** It uses distance metrics like Euclidean or Manhattan distance to find neighbors.

- KNN is intuitive and effective for low-dimensional data but can be computationally expensive for large datasets.

-----------------------------------------------------------------------------------------------------------------------------
## **Decision Trees**
<img width="715" height="450" alt="image" src="https://github.com/user-attachments/assets/7f821549-43ad-40d0-a3ec-1e18774d6eed" />

- Decision Trees use a tree-like structure to make predictions by learning simple decision rules from data features.
- Each node represents a feature test, each branch a decision outcome, and each leaf a predicted label or value.
- They are interpretable and flexible, but can overfit if not properly pruned or regularized.


-----------------------------------------------------------------------------------------------------------------------------

## **Training and Testing Data Sets (Train, Validation, Test)**
<img width="580" height="326" alt="image" src="https://github.com/user-attachments/assets/f1253b0d-d116-43ea-bb66-146d99ea4333" />

- **Training set:** Used to teach the model patterns in the data.
- **Validation set:** Used to fine-tune hyperparameters and assess model performance during training.
- **Test set:** Used only once to evaluate final model performance and generalization ability.

## **Loss Functions and Hyperparameters**
- A **loss function** measures the error between predicted and actual values; common ones include Mean Squared Error for regression and Cross-Entropy for classification.
- **Hyperparameters** are external configuration values (like K in KNN or max depth in DT) that guide the learning process but aren't learned from data.

## **Hyperparameter Optimization**
The process of finding the best set of hyperparameters to improve model performance.
- Techniques include grid search, random search, and cross-validation.
- Effective tuning helps balance model complexity and generalization.

-----------------------------------------------------------------------------------------------------------------------------
## **Distance Functions: Euclidean Distance, Manhattan Distance, and Mahalanobis Distance**
<img width="544" height="246" alt="image" src="https://github.com/user-attachments/assets/4c495d02-55b2-4316-9ba9-655bff83eae6" />

- **Euclidean Distance:** Straight-line distance between two points in space; commonly used in KNN. (L1)
- **Manhattan Distance:** Sum of absolute differences along each dimension; useful for grid-like data. (L2)

- **Additional --> Mahalanobis Distance:** Accounts for correlations between variables and is scale-invariant.


-----------------------------------------------------------------------------------------------------------------------------
## **Using Scikit-Learn for Regression Modelling in ML**
Scikit-learn provides a consistent API for training and evaluating models.

- For regression analysis, use these functions -->  **LinearRegression, DecisionTreeRegressor, or KNeighborsRegressor.**
- Fit models using these functions --> **.fit(), predict with .predict(),** 
- Evaluate models after fitting using these metrics -->  **R², MSE, or MAE.**

## **Common ML Challenges: Underfitting vs. Overfitting**
- **Underfitting:** The model is too simple and fails to capture patterns, resulting in poor training and testing performance.
- **Overfitting:** The model learns noise and specifics of training data too well, leading to poor generalization on unseen data.
**Key Takeaway:** A good model will have a good balance and will perform well on both training and test data.

-----------------------------------------------------------------------------------------------------------------------------

## **Normalization Techniques: Standardization and Min-Max Normalization**
<img width="560" height="361" alt="image" src="https://github.com/user-attachments/assets/bcbec63d-a96b-4748-9c9b-442d70087521" />

- **Standardization:** Scales features to have a mean of 0 and standard deviation of 1.
- **Min-Max Normalization:** Scales features to a fixed range, typically [0, 1].

**Key Takeaway:** Normalization helps models converge faster and improves distance-based model performance.

-----------------------------------------------------------------------------------------------------------------------------

## **Entropies in Information Theory**
- In information theory, entropy is a measure of uncertainty or randomness associated with a random variable. 
- It quantifies the average amount of information needed to describe the outcome of that variable. 
  - Mainly, it tells us how unpredictable a situation is. 


## **Low Entropy vs. High Entropy**
<img width="1100" height="513" alt="image" src="https://github.com/user-attachments/assets/b88a6627-36e4-4079-9236-a3940e5adf45" />

- **Low Entropy:** Indicates low uncertainty or predictability. A variable with low entropy has a few likely outcomes or one outcome that is much more probable than others.
- **High Entropy:** Indicates high uncertainty or randomness. A variable with high entropy has many possible outcomes with relatively even probabilities.

-----------------------------------------------------------------------------------------------------------------------------

## **Model Complexity: The Bias-Variance Trade Off**
<img width="492" height="309" alt="image" src="https://github.com/user-attachments/assets/33ee2733-0da4-4b71-9e4b-e72a1be02709" />


- **Bias:** Error due to overly simplistic assumptions in the model.
- **Variance:** Error due to sensitivity to small fluctuations in the training set.

**Key Takeaway:** A well-tuned model minimizes both, balancing complexity to avoid underfitting and overfitting.


-----------------------------------------------------------------------------------------------------------------------------

# Unit 4: Train a Linear Model 


















