# AI-Based Student Performance Prediction and Early Academic Risk Detection

A complete educational deep-learning project demonstrating:
- NumPy arrays, reshaping, slicing and dot products
- Perceptron
- Sigmoid, Tanh and ReLU visualization
- Manual 2-layer neural network
- Forward propagation and backpropagation
- MSE and Cross Entropy
- SGD and Adam concepts
- TensorFlow/Keras Multilayer Perceptron
- Accuracy, Precision, Recall, F1-score and confusion matrix
- Streamlit prediction dashboard
- Academic risk detection and recommendations

## Project structure

Student_Performance_AI/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── student_performance.csv
├── models/
├── results/
├── src/
│   ├── numpy_demo.py
│   ├── perceptron.py
│   ├── activations.py
│   ├── manual_nn.py
│   ├── losses_optimizers.py
│   ├── train_model.py
│   └── predict.py
└── notebooks/

## Windows setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Train

python train.py

## Run dashboard

streamlit run app.py

If the model files do not exist, the dashboard offers a training button.

## Important

The final_score column is used only to create the target performance category.
It is NOT used as a model input, preventing target leakage.

For real deployment, use a sufficiently large, representative, ethically
collected dataset and validate the model before making academic decisions.
