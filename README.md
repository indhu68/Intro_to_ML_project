

# Age group Prediction from Voice Data  - Intoduction to Machine learning research project

## Project Overview  
This project focuses on predicting the **age group** of a speaker using **voice data**. The methodology involves both **classical machine learning (ML)** and **neural network-based approaches** to analyze audio features and predict the age category. The study evaluates and compares various models, including **Logistic Regression, Support Vector Machines (SVC), K-Nearest Neighbors (KNN), Feedforward Neural Networks (FNN), and Convolutional Neural Networks (CNN)**.

This project was developed as part of the **ECGR 5105 - Machine Learning** coursework at the **University of North Carolina Charlotte**.

📌 **GitHub Repository:** https://github.com/indhu68/Age-Group-Prediction-using-Voice-ML-data_Intro_to_ML_research_project

---

## Introduction  
Voice carries hidden patterns that reveal a speaker’s **age, emotion, and identity**. Estimating age from voice has applications in **security, forensic analysis, AI-driven personalization, and healthcare**. This project explores how well different **machine learning and deep learning models** can predict age groups based on extracted **audio features**.

By implementing **classical ML models** and **deep learning architectures**, we compare their effectiveness and gain insights into the most suitable approach for age prediction from voice recordings.

---

## Approach & Methodology  

### **1️⃣ Classical Machine Learning Models**  
For traditional ML, we extract **23 audio features**, including:  
- **Spectral Features**: Spectral Centroid, Spectral Bandwidth, Spectral Rolloff  
- **Mel-frequency Cepstral Coefficients (MFCCs)**: mfcc1 to mfcc20  
- **Other Features**: Chroma, Spectral Contrast, Tonnetz, RMS Energy  

Using these features, we train and evaluate:  
- **Logistic Regression** – Establishes a baseline for classification.  
- **Support Vector Classifier (SVC) with RBF Kernel** – Captures complex, non-linear relationships in voice data.  
- **K-Nearest Neighbors (KNN)** – Works well for localized pattern recognition.

---

### **2️⃣ Neural Network Models**  
For deep learning, we extract **191 audio features** to capture intricate patterns in voice recordings. We implement:  
- **Feedforward Neural Network (FNN)** – Uses multiple dense layers to model age-related patterns.  
- **Convolutional Neural Network (CNN)** – Utilizes **residual blocks** to extract hierarchical relationships from spectrograms.

---

## Dataset & Feature Extraction  
- **Dataset:** Mozilla Common Voice Dataset ([Kaggle](https://www.kaggle.com/))  
- **Feature Extraction:** `Librosa` Python library for extracting **MFCCs, spectral features, chroma, and energy-based attributes**.

We process the dataset by:  
- Splitting it into **training (80%) and testing (20%)** datasets.  
- Encoding the labels using `LabelEncoder` from `scikit-learn` for ML models.  
- Converting data to `PyTorch` tensors for deep learning models.  

---

## Results & Model Comparison  

|            **Model**               | **Accuracy** | **F1 Score (Macro Avg.)** | **F1 Score (Weighted Avg.)** |
|------------------------------------|--------------|---------------------------|------------------------------|
| Logistic Regression                | 36%          | 16%                       | 29%                          |
| Support Vector Classifier (SVC)    | 78%          | 80%                       | 79%                          |
| K-Nearest Neighbors (KNN)          | **85%**      | **85%**                   | **85%**                      |
| Feedforward Neural Network (FNN)   | 72.70%       | 75%                       | 73%                          |
| Convolutional Neural Network (CNN) | 73%          | 78%                       | 73%                          |

### **Observations:**  
- **KNN outperformed all models**, achieving the highest accuracy (**85%**).  
- **SVC with RBF Kernel** performed well with **78% accuracy**, indicating its effectiveness in handling non-linear relationships.  
- **Neural Networks (FNN & CNN)** performed **competitively**, highlighting their ability to capture intricate voice characteristics.  
- **Logistic Regression struggled (36% accuracy)**, indicating its limitations in handling complex voice features.

---

## Conclusion  
This project demonstrates that **machine learning can effectively predict age groups from voice data**. **KNN and SVC models** show promising results, while **deep learning models can be further optimized** for better performance. The findings contribute to **security applications, forensic analysis, and AI-based personalization**.

🚀 **Next Steps:**  
- Experiment with **more advanced deep learning models (e.g., Transformers for audio processing).**  
- Fine-tune **hyperparameters** and **feature selection techniques**.  
- Train models on **larger and more diverse datasets** for improved generalization.

---

## Authors  
- **Indhuja Gudluru** – Masters in Computer Engineering, UNC Charlotte  
- **Sai Krishna Reddy Mareddy** – Masters in Computer Engineering, UNC Charlotte
- **Poojitha Rajapuram** – Masters in Computer Engineering, UNC Charlotte  
- **Ruby Manderna** – Masters in Computer Engineering, UNC Charlotte  




