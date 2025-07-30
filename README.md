# Break-Through-Tech-Portfolio
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/dc25a42d-cc2e-4667-a7c0-1a8d5b2ae2ef" />

- My Data Science, Machine Learning, and Artificial Intelligence Portfolio developed through Cornell University's Break Through Tech Fellowship program (2025-2026). 

-----------------------------------------------------------------------------------------------------------------------------

# About Me Section

👋 Hi, I'm Kimberly Hernandez!
I'm a junior at the University of Maryland Global Campus, pursuing a double major in Data Science and Finance. I’m passionate about using data-driven insights to solve real-world problems in finance, business, tech, and nonprofit sectors. My interests lie at the intersection of machine learning, financial markets, and technology—areas I explore through academic projects, applied research, and fellowships like Break Through Tech AI. 

[🔗 Connect with me on Linkedin!](https://www.linkedin.com/in/kimberly-h-b91555269/)

# Projects Completed During My ML Foundations Course:

## **Lab 8 Project**
[Lab 8 -- Define And Solve ML Problems Using a Book Review Dataset.pdf](https://github.com/user-attachments/files/21513864/Lab.8.--.Define.And.Solve.ML.Problems.Using.a.Book.Review.Dataset.pdf)

- **Objective:** Built a binary classification model to predict whether a book review is positive or negative using Natural Language Processing (NLP).
- **Dataset:** Used a labeled dataset of book reviews with textual data and sentiment labels (True/False).
- **Methods:** Applied text preprocessing (cleaning, tokenization, lemmatization, stopword removal), TF-IDF vectorization, and trained multiple models including Logistic Regression, Naive Bayes, and SVM.
- **Results:** Achieved an 85% accuracy score using Logistic Regression with hyperparameter tuning via GridSearchCV, validated by precision, recall, and F1-score metrics.

## **Lab 7 Project: Implementing a Convolutional Neural Network Using Keras**
_**(Project is uploaded in repository)**_
- **Objective:** Implemented a Convolutional Neural Network (CNN) for image classification using the Fashion MNIST dataset.
- **Approach:** Built a deep learning pipeline using TensorFlow/Keras, including data normalization, model design, training, and evaluation.
- **Model Design:** Created a CNN with convolutional, max-pooling, and dense layers, achieving high training and validation accuracy.
- **Results:** Evaluated model performance using accuracy and loss plots, confusion matrix, and classification report to validate classification quality.

-----------------------------------------------------------------------------------------------------------------------------
# Summary of Course Learning Content
-----------------------------------------------------------------------------------------------------------------------------
# Unit 1: Machine Learning in a Nutshell (Introduction)
- Understand the machine learning life cycle and explore common machine learning packages. 

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
<img width="700" height="425" alt="image" src="https://github.com/user-attachments/assets/bd923f7d-7141-4295-b86c-4ee37e7ea2ec" />

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
<img width="600" height="320" alt="image" src="https://github.com/user-attachments/assets/d80fd45c-8937-45a6-8b14-a29ce15f4098" />

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
<img width="600" height="430" alt="image" src="https://github.com/user-attachments/assets/20f4719b-c6c1-4c47-9e18-e03dbf5eeb1e" />

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
<img width="425" height="300" alt="image" src="https://github.com/user-attachments/assets/7f821549-43ad-40d0-a3ec-1e18774d6eed" />

- Decision Trees use a tree-like structure to make predictions by learning simple decision rules from data features.
- Each node represents a feature test, each branch a decision outcome, and each leaf a predicted label or value.
- They are interpretable and flexible, but can overfit if not properly pruned or regularized.


-----------------------------------------------------------------------------------------------------------------------------

## **Training and Testing Data Sets (Train, Validation, Test)**
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f1253b0d-d116-43ea-bb66-146d99ea4333" />

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
<img width="425" height="300" alt="image" src="https://github.com/user-attachments/assets/4c495d02-55b2-4316-9ba9-655bff83eae6" />

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
<img width="450" height="350" alt="image" src="https://github.com/user-attachments/assets/bcbec63d-a96b-4748-9c9b-442d70087521" />

- **Standardization:** Scales features to have a mean of 0 and standard deviation of 1.
- **Min-Max Normalization:** Scales features to a fixed range, typically [0, 1].

**Key Takeaway:** Normalization helps models converge faster and improves distance-based model performance.

-----------------------------------------------------------------------------------------------------------------------------

## **Entropies in Information Theory**
- In information theory, entropy is a measure of uncertainty or randomness associated with a random variable. 
- It quantifies the average amount of information needed to describe the outcome of that variable. 
  - Mainly, it tells us how unpredictable a situation is. 


## **Low Entropy vs. High Entropy**
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/b88a6627-36e4-4079-9236-a3940e5adf45" />

- **Low Entropy:** Indicates low uncertainty or predictability. A variable with low entropy has a few likely outcomes or one outcome that is much more probable than others.
- **High Entropy:** Indicates high uncertainty or randomness. A variable with high entropy has many possible outcomes with relatively even probabilities.

-----------------------------------------------------------------------------------------------------------------------------

## **Model Complexity: The Bias-Variance Trade Off**
<img width="425" height="300" alt="image" src="https://github.com/user-attachments/assets/33ee2733-0da4-4b71-9e4b-e72a1be02709" />


- **Bias:** Error due to overly simplistic assumptions in the model.
- **Variance:** Error due to sensitivity to small fluctuations in the training set.

**Key Takeaway:** A well-tuned model minimizes both, balancing complexity to avoid underfitting and overfitting.


-----------------------------------------------------------------------------------------------------------------------------
# Unit 4: Train a Linear Model 
- Learn how to apply gradient descent, loss functions, and hyperparameters for logistic regression.
- Use NumPy for vector and matrix operations for ML problems. 

-----------------------------------------------------------------------------------------------------------------------------

# Key Concepts Covered:

## **Dot Product (Scalar Product)**
- Also referred to as scalar product, a special operation that takes the element’s product of each vector and then sums them together.
  - It is typically represented with what looks like a period or a dot floating between the two vectors.

-----------------------------------------------------------------------------------------------------------------------------

## **Gradient Descent**
- A numerical optimization algorithm used to train and optimize a logistic regression model using loss functions.
  - **Gradient Descent** iteratively updates the model parameters until a loss function is minimized. 

-----------------------------------------------------------------------------------------------------------------------------

## **Hessian and Identity Matrices**

- **Hessian Matrix:** A square matrix of second-order partial derivatives of a scalar-valued function, or scalar field.
- **Identity Matrix:** A special matrix whose diagonal elements are one, and all other elements are zero.


-----------------------------------------------------------------------------------------------------------------------------
## **An In-Depth Guide to Logistic Regression Models**

- **Logistic Regression:** A linear classification method that is trained by iteratively tuning a set of weights to minimize the log loss.

## **Inverse Logit**
- One of three steps in the logistic regression model. The inverse logit step transforms the output of the linear step into a probability prediction P(y|X) between 0-1.

## **The Learning Rate Hyperparameter in Logistic Regression**
- A common logistic regression hyperparameter (also commonly known as the step size), learning rate dictates the speed of gradient descent. The ideal learning rate is one that reaches global minima in a fast and efficient manner.

## **Linear Step**
- One of three steps in the Logistic Regression model. The linear step computes a value z(X) by taking the linear sum of feature values z (X with model weights) W and an intercept term α (also known as bias, β, in some literature).

## **Log Loss**
- A popular loss function used to measure the performance of a some classification models. This loss function is used for both logistic regression and neural networks.

## **What is a Linear Model?**
- A class of supervised learning models that are represented by an equation: simple to implement, fast to train, and have lower complexity.
  - Logistic regression is an example of a linear model.
- In a linear model, the form of a model must fit a very specific format. 

-----------------------------------------------------------------------------------------------------------------------------

## **An In-Depth Guide to Linear Regression Models**
<img width="500" height="475" alt="image" src="https://github.com/user-attachments/assets/0abac555-42fd-4c2c-8330-2fb93ef965eb" />

- **Linear Regression** is supervised machine learning algorithm used for regression problems. It finds a linear relationship between one or more features and a label (such as a price or an age.)
- **There are two types of linear regression models:** **Simple linear regression** finds the linear relationship between one feature and one label, and **multiple linear regression** finds the linear relationship between multiple features and one label.

-----------------------------------------------------------------------------------------------------------------------------

## **Mean Squared Error (MSE)**
<img width="500" height="375" alt="image" src="https://github.com/user-attachments/assets/5983a400-f526-4a8d-a4b0-054b059d0b85" />

- Mean Squared Error (MSE) is a common loss function used in regression tasks to measure the average squared difference between the predicted values and the actual labels.
- It is calculated by squaring the error (difference between predicted and actual), summing all squared errors, and then dividing by the total number of data points.

**Key Takeaways of MSE:**
- A lower MSE indicates better model performance and more accurate predictions.
- **Note:** It penalizes larger errors more heavily than smaller ones, making it sensitive to outliers.

## **Regularization Penalty**
- The penalty on a model’s complexity; which helps to prevent overfitting. Different kinds of regularization include L1 regularization (Lasso Regression) and L2 regularization (Ridge Regression).

-----------------------------------------------------------------------------------------------------------------------------

# Unit 5: Evaluate and Deploy Your Model
- Understand the importance of selection in ML --> to select the right model evaluation metrics that are appropriate for the analysis type. 
- Choose appropriate model candidates and hyperparameters for training/validation/test splits in the selection phase. 
- Apply feature selection techniques to deploy, host, and monitor your model

-----------------------------------------------------------------------------------------------------------------------------

# Key Concepts Covered:

## **Feature Selection**
- **Feature selection** is the process of identifying the most relevant input variables (features) that contribute to the model’s performance.
- It helps to improve model accuracy, reduce overfitting, and decrease training time by removing irrelevant or redundant data.

## **Heuristic Selection**
- **Heuristic selection** refers to rule-of-thumb strategies or approximations used to guide feature selection when exact methods are computationally expensive.

## **Stepwise Selection: Forward vs. Backward**
- **Stepwise Selection** is a systematic method of adding or removing features based on their statistical significance in a regression model.

- **Forward Stepwise** selection begins with no features and adds them one at a time.
- **Backward Stepwise** selection starts with all features and removes the least significant one at each step.

-----------------------------------------------------------------------------------------------------------------------------
## **Precision vs. Accuracy**
<img width="636" height="340" alt="image" src="https://github.com/user-attachments/assets/9b73178e-36fb-413b-9f33-95aa29264285" />


- **Precision** focuses only on positive predictions, calculating the proportion of true positives among all predicted positives.
- **Accuracy** measures the overall correctness of the model by comparing all correct predictions to total predictions.

**Key Takeaway:** High accuracy can be misleading with imbalanced datasets; precision is more informative when false positives are costly.

-----------------------------------------------------------------------------------------------------------------------------
## **Hyperparameters**

- **Hyperparameters** are external settings that control how a model learns, such as learning rate, tree depth, or number of neighbors in KNN.
- Unlike model parameters (which are learned from data), hyperparameters are set before training and affect model performance.
  - Tuning hyperparameters is essential for optimizing model accuracy, generalization, and training efficiency.

-----------------------------------------------------------------------------------------------------------------------------
## **Cross-Validation in ML**
- **Cross-validation** is a technique for evaluating model performance by partitioning data into multiple training and testing splits.
  - It reduces the risk of overfitting by providing a more reliable estimate of how the model will perform on unseen data. 


## **K-Fold Cross Validation**
<img width="420" height="569" alt="image" src="https://github.com/user-attachments/assets/7dd8610c-e43b-4f1e-a2ea-d086078488cd" />

- In **K-Fold Cross Validation**, the dataset is split into K equal parts (folds). The model is trained on K-1 folds and tested on the remaining fold. This process repeats K times, with each fold serving as the test set once.
  - The final performance metric is the average of all K test scores.

-----------------------------------------------------------------------------------------------------------------------------
## **Receiver Operator Curve (ROC)**
- The **ROC Curve** is a graphical plot that illustrates a binary classifier’s performance by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various thresholds.
- A model with perfect classification has a curve that hugs the top-left corner and an **AUC (Area Under Curve)** of 1.0.

-----------------------------------------------------------------------------------------------------------------------------

# Unit 6: Special Topics -- Ensemble Methods and Unsupervised Learning
- Learn about unsupervised learning methods (i.e., unsupervised clustering) and understand the different ensemble methods that can be used to improve model performance.


-----------------------------------------------------------------------------------------------------------------------------

# Key Concepts Covered:

## **Types of Ensemble Methods Used in Unsupervised Learning**
<img width="914" height="376" alt="image" src="https://github.com/user-attachments/assets/1118a203-e0c6-498c-b8b6-04fa13167b34" />


- There are three types of ensemble methods --> (1) Stacking, (2) Bagging, and (3) Boosting. 

## **Stacking**
- Taking the weighted combination of the predictions of a total of K different methods. 

## **Bagging (Bootstrapped Aggregation)**
- Generates multiple models from the same data by taking boostrapped samples and averaging the individual model preductions. 

## **Boosting**
- Iteratively build models by focusing on the cumulative errors from prior iterations' predictions. 

-----------------------------------------------------------------------------------------------------------------------------
## **Model Error Equation**

**Model Error = Estimation Bias + Estimation Variance**

- **Estimation Bias:** Model rigidity that prevents adaptation to nuances of the data. 
- **Estimation Variance:** Model flexibility that causes the estimated model to be sensitive to data nuances

**Note: This equation for model error is used in ensemble models.**

-----------------------------------------------------------------------------------------------------------------------------
## **Random Forest Algorithm** 
- Random Forest is an ensemble learning method that constructs a collection of decision trees during training and outputs the average prediction (regression) or majority vote (classification).

**It introduces randomness by:**
- Bootstrapping the dataset (sampling with replacement).
- Selecting random subsets of features at each split.

**Advantages:**
- Reduces overfitting compared to individual decision trees.
- Handles missing data and maintains good accuracy without extensive tuning.

-----------------------------------------------------------------------------------------------------------------------------
## **Gradient Boosted Decision Trees**
- **Gradient Boosted Decision Trees (GBDT)** build trees sequentially, where each new tree corrects the residual errors made by the previous ones.
- The process uses gradient descent to minimize a specified loss function (e.g., MSE or log loss).

**Disadvantage:** Can be sensitive to overfitting if not properly tuned (e.g., learning rate, number of trees, depth).

-----------------------------------------------------------------------------------------------------------------------------
## **Unsupervised Learning Algorithms**
- Unsupervised learning methods are used to find patterns or structures in data without using labeled outputs.

## **K-Means Clustering**
<img width="561" height="420" alt="image" src="https://github.com/user-attachments/assets/358e18a5-da98-466d-aa5e-7b929d9ac93d" />

- **K-Means** is an unsupervised algorithm that clusters data by minimizing the distance between each data point and its assigned cluster center.
  - **Key Consideration:** Initialization and choice of K can affect results. The elbow method is commonly used to select K.

## **Heirarchical Clustering**
<img width="600" height="460" alt="image" src="https://github.com/user-attachments/assets/25f181c7-e0c7-46a3-bc5e-5c0e4dcbf6ea" />

- **Hierarchical Clustering** builds a hierarchy of clusters that can be visualized using a dendrogram.

**There are Two Main Types:**
- **Agglomerative (bottom-up):** Starts with individual points and merges the closest clusters step by step.
- **Divisive (top-down):** Starts with one cluster and splits it iteratively.

  - **Advantages:** Useful for exploring the structure of data and making decisions about cluster boundaries after visualization.


-----------------------------------------------------------------------------------------------------------------------------

# Unit 7: Special Topics -- Deep Learning and Computer Vision

- Explore the architecture and logic behind Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs), commonly used for image classification tasks.


-----------------------------------------------------------------------------------------------------------------------------

## Neural Networks (NN)
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/5578ab4e-6724-42cc-a9bd-b56bdeae5d11" />

- **Neural Networks** are composed of interconnected layers of nodes (neurons), each performing weighted computations followed by an activation function.

- Training a neural network involves minimizing a **log loss function (cross-entropy),** which penalizes misclassifications by computing the negative log probability of the correct class.

-----------------------------------------------------------------------------------------------------------------------------

## Entropy and Log Loss

- **Entropy** measures uncertainty in classification.
- The **log loss function** uses **entropy** to penalize incorrect predictions—lower log loss means better performance.

-----------------------------------------------------------------------------------------------------------------------------

## Convolutional Neural Networks (CNNs)
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/cd578b24-7704-4faa-bc90-d76000624d02" />

## Convolution Layers

- CNNs extract spatial features from images using filters (kernels) that slide over the image, capturing patterns like edges or textures.

- The **convolution operation generates a feature map**, reducing the image into its most essential information.

## Pooling Layers

- **Max pooling** and **average pooling** reduce the spatial dimensions of the feature map, helping with generalization and computational efficiency.

- **Example:** A 4×4 region may shrink to 2×2 after pooling, preserving critical features.

## Batch Normalization

- Applied during training to normalize intermediate outputs of layers.
- It speeds up convergence, stabilizes learning, and allows for higher learning rates. 

-----------------------------------------------------------------------------------------------------------------------------


# Unit 8: Special Topics -- Deep Learning and Natural Language Processing

- Explore how deep learning techniques are used to understand, process, and generate human language, with applications ranging from sentiment analysis to large language models.

-----------------------------------------------------------------------------------------------------------------------------

## What is Natural Language Processing (NLP)? 
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/a20cc188-0528-4da9-8eb6-d6f35f03c4c2" />

- NLP is a branch of Artificial Intelligence (AI) that enables machines to interpret, analyze, and generate human language in text or speech.

- It powers systems like chatbots, voice assistants, translation tools, and content recommendation engines.


-----------------------------------------------------------------------------------------------------------------------------

## The NLP Pipeline Process

**Data Cleaning and Preprocessing**
- Remove stop words (e.g., and, is, the)
- Apply tokenization (split text into words or tokens)

**Vectorization**
- Convert words or documents into numerical vectors
- Methods include TF-IDF and word embeddings

**Modeling**
- Apply machine learning or deep learning models to classify, cluster, or generate text


-----------------------------------------------------------------------------------------------------------------------------

## Word Embeddings 

- Represent words as K-dimensional vectors (typically within a -1 to 1 range)
- Embeddings capture semantic meaning and contextual similarity between words


## Large Language Models (LLMs)
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/4484df4a-bbae-4fee-b0fd-0b33cd5a1377" />

- LLMs like GPT and BERT are pre-trained on massive text corpora and can understand or generate human-like language. They power state-of-the-art applications in translation, summarization, Q&A systems, and more. 


-----------------------------------------------------------------------------------------------------------------------------

# Unit 9: Prepare ML Models for the Real World

- Explore the practical and ethical considerations for deploying machine learning models at scale, including performance issues, class imbalance, and fairness.

-----------------------------------------------------------------------------------------------------------------------------

## Solution Engineering

- **Model performance failure:** When evaluating a test, a performance failure describes a model that may have poor performance. This can lead to overconfidence in a system and poor generalization performance when the model is applied to live data.

- **Execution bottleneck:** A data preparation or modeling process not terminating in a reasonable amount of time. This common and frustrating issue is often caused by a misalignment among the data, tools, and hardware you use to process the data.

- **Societal failures:** Failures which happen when a machine learning model produces unintended discrimination or disparate impact and lacks accountability. It is where failure exists between the model’s owner and the social context in which the model operates.

-----------------------------------------------------------------------------------------------------------------------------

## Data Sufficiency and Class Imbalance

## Class Imbalance

- A common issue in supervised learning when one class significantly outweighs another (e.g., 90% negative, 10% positive).
- Can lead to biased models that favor the majority class.

## Applicable Solutions

- **Downsampling:** Reduces the size of the majority class.
- **Upsampling:** Increases the size of the minority class using duplication or synthesis.
- These techniques aim to rebalance training data and improve model sensitivity to underrepresented outcomes.

-----------------------------------------------------------------------------------------------------------------------------

## Feature Engineering Challenges

- **Incomplete Features:** May cause poor model accuracy and limit generalization.
- **Improved Feature Selection:** Helps focus on the most relevant variables, often informed by domain knowledge and importance metrics.

-----------------------------------------------------------------------------------------------------------------------------

## Fairness and Accountability in Machine Learning

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/ce66c148-d11a-401c-afb1-06bd207edebc" />


**Addressing ethical AI concerns means ensuring that ML systems:**
1. Avoid discrimination or disparate impact,
2. Are explainable and transparent,
3. Operate responsibly in social and legal contexts.

**Tools like fairness metrics and bias audits can support equitable ML deployment.**


-----------------------------------------------------------------------------------------------------------------------------

