# Credit-Card-Fraud-Detection-with-Decision-Trees-and-SVM
Credit Card Fraud Detection with Decision Trees and SVM
#Dataset: "https://www.kaggle.com/mlg-ulb/creditcardfraud"

# **Comprehensive Guide to Support Vector Machines (SVMs) in Supervised Learning**  

## **1. Introduction to Support Vector Machines (SVMs)**  
Support Vector Machines (SVMs) are a powerful **supervised learning** algorithm used for **classification** and **regression** tasks. They work by finding the optimal **decision boundary (hyperplane)** that best separates different classes in the dataset.  

### **1.1 Key Characteristics of SVMs**  
- **Maximizes Margin**: SVMs aim to find the hyperplane that maximizes the distance (margin) between the closest points of different classes.  
- **Works in High-Dimensional Spaces**: Effective even when the number of features exceeds the number of samples.  
- **Kernel Trick**: Can handle non-linear decision boundaries by transforming data into higher dimensions.  
- **Robust to Overfitting**: Particularly effective when the number of features is large compared to the number of observations.  

---

## **2. How SVMs Work**  

### **2.1 Linear SVM (Hard Margin SVM)**  
- **Objective**: Find the hyperplane that perfectly separates two classes with the maximum margin.  
- **Mathematical Formulation**:  
  - Given training data \( (x_i, y_i) \), where \( y_i \in \{-1, 1\} \), SVM finds \( w \) (weight vector) and \( b \) (bias) such that:  
    \[
    y_i (w \cdot x_i + b) \geq 1 \quad \forall i
    \]
  - The margin width is \( \frac{2}{\|w\|} \), so maximizing the margin is equivalent to minimizing \( \|w\| \).  

### **2.2 Soft Margin SVM (Handling Non-Separable Data)**  
- **Problem**: Real-world data is often noisy and not perfectly separable.  
- **Solution**: Introduce a **slack variable** \( \xi_i \) to allow some misclassifications.  
  \[
  \text{Minimize} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i
  \]
  \[
  \text{Subject to} \quad y_i (w \cdot x_i + b) \geq 1 - \xi_i \quad \text{and} \quad \xi_i \geq 0
  \]
  - **C (Regularization Parameter)**:  
    - **Small C**: Wider margin, more misclassifications allowed (softer margin).  
    - **Large C**: Narrower margin, fewer misclassifications (harder margin).  

### **2.3 Non-Linear SVM (Kernel Trick)**  
- **Problem**: Some datasets are not linearly separable.  
- **Solution**: Map data to a higher-dimensional space where separation is possible.  

#### **Common Kernel Functions**  
| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | \( K(x_i, x_j) = x_i \cdot x_j \) | Linear decision boundaries |
| **Polynomial** | \( K(x_i, x_j) = (x_i \cdot x_j + c)^d \) | Non-linear boundaries |
| **RBF (Gaussian)** | \( K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2) \) | Complex non-linear boundaries |
| **Sigmoid** | \( K(x_i, x_j) = \tanh(\alpha x_i \cdot x_j + c) \) | Neural network-like behavior |

---

## **3. Support Vector Regression (SVR)**  
SVMs can also be used for **regression** by modifying the objective function.  

### **3.1 Key Concepts in SVR**  
- **Epsilon (ε)-Insensitive Tube**: Predictions within \( \pm \epsilon \) of the true value are not penalized.  
- **Objective**: Minimize the deviation outside the ε-tube while keeping the model as flat as possible.  
  \[
  \text{Minimize} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
  \]
  \[
  \text{Subject to} \quad 
  \begin{cases}
  y_i - (w \cdot x_i + b) \leq \epsilon + \xi_i \\
  (w \cdot x_i + b) - y_i \leq \epsilon + \xi_i^* \\
  \xi_i, \xi_i^* \geq 0
  \end{cases}
  \]

### **3.2 Example of SVR with Different ε Values**  
- **Small ε**: Tight fit, fewer points inside the tube (more sensitive to noise).  
- **Large ε**: Wider tube, more points considered correct (more tolerant to noise).  

---

## **4. Applications of SVM**  

### **4.1 Classification Use Cases**  
✔ **Image Recognition** (e.g., handwritten digit classification)  
✔ **Spam Detection** (classifying emails as spam or not)  
✔ **Sentiment Analysis** (determining positive/negative reviews)  
✔ **Bioinformatics** (gene classification, cancer detection)  

### **4.2 Regression Use Cases**  
✔ **Stock Market Prediction**  
✔ **Weather Forecasting**  
✔ **Medical Diagnosis** (predicting disease progression)  

---

## **5. Advantages and Limitations of SVM**  

### **5.1 Advantages**  
✅ **Effective in High Dimensions**: Works well even when features > samples.  
✅ **Robust to Overfitting**: Due to margin maximization.  
✅ **Flexible Kernel Choices**: Can model non-linear relationships.  
✅ **Memory Efficient**: Only support vectors are needed for prediction.  

### **5.2 Limitations**  
❌ **Computationally Expensive**: Training time increases with dataset size.  
❌ **Sensitive to Parameter Tuning**: Poor choice of \( C \), \( \epsilon \), or kernel can hurt performance.  
❌ **Not Ideal for Large Datasets**: Better suited for small to medium datasets.  
❌ **Black-Box Nature**: Hard to interpret compared to decision trees.  

---

## **6. Implementing SVM in Python (Scikit-Learn)**  

### **6.1 SVM for Classification (SVC)**  
```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train SVM with RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### **6.2 SVM for Regression (SVR)**  
```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Train SVR with RBF kernel
model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
model.fit(X, y)

# Predict
y_pred = model.predict(X)
```

---

## **7. Conclusion**  
- **SVMs are powerful** for both classification and regression.  
- **Kernel trick** allows handling non-linear data.  
- **Parameter tuning (C, ε, kernel choice)** is crucial for performance.  
- **Best suited for medium-sized datasets** where interpretability is not the main concern.  
