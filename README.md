# Heart Disease Prediction using Keras

## Objective
The goal of this project is to build a binary classification model using Keras to predict the likelihood of heart disease based on various health indicators. The model aims to achieve an accuracy of +93%, but the current model's accuracy is 90.9%.

## Dataset Description
The dataset used for this project contains 22 columns, with `HeartDiseaseorAttack` as the target variable. The columns represent various health indicators such as:
- **HighBP**: High blood pressure status
- **BMI**: Body mass index
- **Smoker**: Smoking status
- **Diabetes**: Diabetes status
- ... and more health-related features.

The target variable, `HeartDiseaseorAttack`, is a binary indicator representing whether the individual has experienced heart disease or a heart attack (1) or not (0).

## Project Steps

### 1. Data Preprocessing
   - **Load the Dataset**: The dataset is read into a pandas DataFrame and split into features (`X`) and target (`y`).
   - **Normalize the Features**: Standardize the input features to ensure all inputs to the neural network are on a similar scale.
   - **Split the Data**: The dataset is divided into training and testing sets (80% training, 20% testing) to evaluate the model.

### 2. Model Creation
   - **Build the Neural Network**: A neural network is constructed with Keras, consisting of input layers, hidden layers, and an output layer.

### 3. Model Compilation
   - **Compile the Model**: The model is compiled with an optimizer, loss function, and metrics (accuracy) to be used for training.

### 4. Model Training and Evaluation
   - **Train the Model**: The model is trained on the training data, with validation data used to monitor the training process.
   - **Evaluate the Model**:
     - Accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC curve are calculated to evaluate the model's performance.
   - **Visualize the Training Process**:
     - TensorBoard is used to visualize the training and validation loss, helping to assess how well the model is learning over time.

### 5. Hyperparameter Tuning (Bonus)
   - **Use Keras Tuner**: Perform hyperparameter tuning to optimize the number of layers, neurons, and learning rate for better performance.

## Results
The model achieved an accuracy of **90.9%** on the test dataset. Although the initial target was +93%, further hyperparameter tuning and model adjustments may improve performance.

## Dependencies and Installation Instructions
To run this project, you need the following dependencies:
- Python 3.x
- Jupyter Notebook
- Keras
- TensorFlow
- scikit-learn
- matplotlib
- numpy
- pandas

### Installation
You can install the required packages using the following commands:

```bash
pip install tensorflow
pip install keras
pip install scikit-learn
pip install matplotlib
pip install numpy
pip install pandas
```

## Running the Code
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
3. **Run the Notebook**:
   - Open the `.ipynb` file and execute each cell to run the code and see the outputs.
     You might need to install the dependencies first

## Training Process and Evaluation

### 1. Loss Visualization
   - TensorBoard is used to visualize the training and validation loss over time. Ideally, both should decrease and converge to show that the model is learning well.

### 2. ROC-AUC Curve
   - The ROC-AUC curve is plotted to assess the model's ability to distinguish between the two classes.

### 3. Confusion Matrix and Classification Report
   - The confusion matrix displays true vs. predicted classifications.
   - The classification report shows Precision, Recall, F1-score, and Accuracy metrics.


## Training Curves Dashboard
![image](https://github.com/user-attachments/assets/2faed645-b61d-4473-b396-060c1aa420ac)

