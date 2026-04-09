# Weather Forecast Prediction with LSTM

## 📌 Project Overview
This project demonstrates how to build and train a Long Short-Term Memory (LSTM) neural network model to predict whether it will rain or not based on various weather features. The model leverages historical weather data to learn patterns and make binary classifications.

## 📊 Dataset
The dataset used in this project is the "Weather Forecast Dataset" from KaggleHub. It contains the following features:
- `Temperature`: Current temperature.
- `Humidity`: Relative humidity.
- `Wind_Speed`: Wind speed.
- `Cloud_Cover`: Percentage of cloud cover.
- `Pressure`: Atmospheric pressure.
- `Rain`: The target variable, indicating whether it rained ('rain') or not ('no rain').

The dataset is loaded into a pandas DataFrame, and the first few rows are inspected.

## ⚙️ Methodology
The project follows a standard machine learning workflow:

### 1. Data Loading and Initial Exploration
- The dataset is downloaded using `kagglehub`.
- Loaded into a pandas DataFrame.
- Basic exploratory data analysis (EDA) is performed, including:
    - Displaying histograms for all numerical features to understand their distributions.
    - Generating a correlation heatmap to visualize relationships between numerical features.
    - Creating box plots to identify outliers in the numerical features.

### 2. Data Preprocessing
- **Target Encoding**: The categorical 'Rain' column is converted into numerical format (`'no rain': 0`, `'rain': 1`).
- **Feature-Target Split**: The dataset is split into features (X) and target (y).
- **Feature Scaling**: `MinMaxScaler` from `sklearn.preprocessing` is used to scale the numerical features to a range between 0 and 1, which is beneficial for neural networks.
- **Data Reshaping for LSTM**: The scaled feature data is reshaped into a 3D array (`(samples, timesteps, features)`) required by LSTM layers.
- **Train-Test Split**: The data is divided into training and testing sets using `train_test_split` with a 80/20 ratio and `random_state=42` for reproducibility.

### 3. Model Building (LSTM)
- A sequential Keras model is built with the following layers:
    - An `LSTM` layer with 64 units, taking the reshaped input.
    - A `Dropout` layer (0.2) to prevent overfitting.
    - A `Dense` output layer with 1 unit and a 'sigmoid' activation function for binary classification.

### 4. Model Training
- The model is compiled using the 'adam' optimizer, 'binary_crossentropy' loss function, and 'accuracy' as a metric.
- An `EarlyStopping` callback is implemented to monitor 'val_loss' with a patience of 10 epochs, restoring the best weights.
- The model is trained for up to 200 epochs with a `batch_size` of 50.

### 5. Model Evaluation
- The trained model is evaluated on the test set to determine its `loss` and `accuracy`.
- The training and validation accuracy over epochs are plotted to visualize the model's learning progress and identify potential overfitting.

## 📈 Results
After training, the LSTM model achieved a test accuracy of approximately **98.00%**. The plots of training and validation accuracy indicate that the model learned effectively and generalized well to unseen data, with early stopping preventing significant overfitting.

## 🛠️ Technologies Used
- **`kagglehub`**: For downloading the dataset.
- **`pandas`**: For data manipulation and analysis.
- **`numpy`**: For numerical operations, especially reshaping data for LSTM.
- **`matplotlib.pyplot`**: For creating static, interactive, and animated visualizations.
- **`seaborn`**: For making statistical graphics attractive and informative.
- **`sklearn.preprocessing.MinMaxScaler`**: For feature scaling.
- **`sklearn.model_selection.train_test_split`**: For splitting data into training and testing sets.
- **`tensorflow.keras`**: For building, compiling, and training the LSTM neural network model.

## ▶️ How to Run the Notebook
1.  **Clone the Repository (if applicable)**:
    ```bash
git clone <repo-link>
cd <repo-name>
    ```
2.  **Install Dependencies**:
    ```bash
pip install -r requirements.txt
    ```
    (Note: `requirements.txt` would contain `kagglehub`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`)
3.  **Open in Google Colab**: Upload the `.ipynb` file to Google Colab.
4.  **Run Cells**: Execute the cells sequentially to download the data, preprocess it, train the model, and evaluate its performance.

## 📂 Project Structure
This project is primarily contained within a single Google Colab notebook, demonstrating the entire workflow from data loading to model evaluation. The dataset is external and downloaded via `kagglehub`.

## 🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests if you have suggestions or improvements.

## 📄 License
This project is open-sourced under the MIT License.
