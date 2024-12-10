# Breast Cancer Prediction using TensorFlow and Keras

This project demonstrates how to build a breast cancer prediction model using TensorFlow and Keras. The model is trained on the Wisconsin Breast Cancer Diagnostic Dataset and can predict whether a tumor is malignant or benign based on its characteristics.

## Dataset

The Wisconsin Breast Cancer Diagnostic Dataset is used for this project. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

## Model

The prediction model is a simple neural network built using Keras. It consists of an input layer, a hidden layer with 20 neurons and ReLU activation, and an output layer with 2 neurons and sigmoid activation. The model is trained using the Adam optimizer and the sparse categorical cross-entropy loss function.

## Usage

1. **Load the dataset:** The dataset is loaded using the `sklearn.datasets.load_breast_cancer()` function.
2. **Preprocess the data:** The data is preprocessed by standardizing the features using the `StandardScaler` from scikit-learn.
3. **Build the model:** The neural network model is built using Keras.
4. **Train the model:** The model is trained on the training data using the `model.fit()` function.
5. **Evaluate the model:** The model is evaluated on the testing data using the `model.evaluate()` function.
6. **Make predictions:** Predictions can be made on new data using the `model.predict()` function.

## Results

The model achieves an accuracy of around 97% on the testing data.

## Requirements

* Python 3.6 or higher
* TensorFlow 2.0 or higher
* Keras 2.3.0 or higher
* scikit-learn 0.22 or higher
* NumPy 1.18 or higher
* Pandas 1.0 or higher
* Matplotlib 3.2 or higher

## Contributing

Contributions to this project are welcome. Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License.
