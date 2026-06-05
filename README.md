For size limitations, I am unable to give the dataset here. But it can be downloaded from kaggle website:

https://www.kaggle.com/datasets/gaurav2022/mobile-health


🧍‍♂️📱 Human Action Detection (Mobile Health)
📌 Project Overview

This project focuses on building a Human Action Recognition system using deep learning techniques for mobile health applications. The system is designed to identify and classify human activities from images or video frames, which can be useful in healthcare monitoring, fitness tracking, and elderly care systems.

The model learns spatial and temporal patterns to recognize different human actions accurately.

🎯 Objective

The main objectives of this project are:

Detect and classify human actions from visual data
Support mobile health monitoring applications
Improve automated activity tracking for healthcare use cases
Build an efficient deep learning-based classification system
📊 Dataset Description

The dataset typically includes:

Images or video frames of human activities
Labels representing different actions (e.g., walking, sitting, standing, running, etc.)
Preprocessed or raw motion-based data

The dataset may be derived from public human activity recognition datasets or custom-collected mobile health data.

🧠 Model Architecture

The project generally uses deep learning models such as:

Convolutional Neural Networks (CNNs) for spatial feature extraction
LSTM / RNN layers for temporal sequence learning (if video-based)
Transfer learning using pre-trained models like MobileNet, ResNet
Softmax classifier for multi-class action prediction
🛠️ Technologies Used
Python 🐍
TensorFlow / Keras or PyTorch
OpenCV
NumPy
Pandas
Scikit-learn
Matplotlib
⚙️ Project Workflow
Data Collection and Labeling
Data Preprocessing (resizing, normalization, augmentation)
Feature Extraction using CNN
Model Training and Validation
Activity Classification
Performance Evaluation
📈 Evaluation Metrics

Model performance is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
🚀 How to Run the Project
1. Clone the repository
git clone https://github.com/souhridkhanra/Human-Action-Detection-mobile-health-.git
2. Navigate to project directory
cd Human-Action-Detection-mobile-health-
3. Install dependencies
pip install -r requirements.txt
4. Run the notebook
jupyter notebook
📌 Applications
Mobile health monitoring 📱
Elderly activity tracking 👵
Fitness and workout analysis 🏋️
Smart surveillance systems 📷
Rehabilitation and medical assistance systems 🏥
📌 Future Improvements
Improve accuracy using transformer-based video models (Vision Transformers)
Deploy real-time mobile app integration
Add pose estimation for better action understanding
Optimize model for edge devices (low-power mobile systems)
Extend dataset for more diverse human activities
📄 License

This project is open-source and available under the MIT License.
