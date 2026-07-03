Guardian Deity — Women & Child Safety Detection Using Audio Signals

An audio signal classification project that detects danger situations involving women and children. By analyzing audio features extracted from .wav files, multiple Machine Learning and Deep Learning models are trained and compared to classify sounds into three categories: Women (distress sounds from women), Child (distress sounds from children), and Normal (everyday ambient sounds).

This project was developed as part of a research paper on real-time danger detection systems.

Project Motivation

The broader vision behind this project is a safety system where, in a danger situation, a person could trigger an alert through a wearable device (pendant, earring, ring) or mobile phone, sending an emergency signal to police or trusted contacts. This notebook focuses on the machine learning component of that vision — building and evaluating models that can automatically detect distress sounds from audio, which could serve as an automatic trigger for such an alert system in the future. The hardware/IoT integration is a future direction and is not part of the implementation in this notebook.

Dataset & Feature Extraction

The dataset consists of .wav audio files labeled into three classes: Women, Child, and Normal. For each audio file, the following features are extracted using librosa:

FeatureDescriptionZCR (Zero Crossing Rate)How often the signal crosses zero — captures noisinessRMS (Root Mean Square Energy)Average energy/loudness of the audioSpectral Centroid"Center of mass" of the spectrum — captures brightnessMFCC 1–13Mel Frequency Cepstral Coefficients — capture tonal/timbral characteristics

The dataset is split into training and testing sets using an 80/20 ratio.

Models Trained & Compared

Five models were trained and evaluated on the extracted features:


Random Forest Classifier (100 trees) — ensemble of decision trees
Logistic Regression — linear baseline model
Gaussian Naive Bayes — probabilistic model based on Bayes' theorem
Decision Tree Classifier — interpretable baseline
TCN (Temporal Convolutional Network) with Attention — a custom deep learning model using dilated causal convolutions with residual connections, plus an attention layer to focus on the most informative parts of the audio sequence


Evaluation

Each model is evaluated using:


Confusion Matrix — visualizes correct vs. incorrect predictions per class
Precision, Recall, and F1-Score — per-class performance metrics
AUC (Area Under the ROC Curve) — using a One-vs-Rest strategy for multi-class evaluation
Radar Plot — a 360° visual comparison of all models across all metrics


Key Takeaways


TCN with Attention achieved the best overall performance, owing to its ability to capture temporal patterns in audio signals that simpler classical ML models cannot.
Classical ML models (Random Forest, Logistic Regression, Naive Bayes, Decision Tree) provide strong, fast baselines and remain competitive for this classification task.
The comparison highlights the trade-off between model complexity (deep learning) and interpretability/speed (classical ML) for real-time safety applications.


Tech Stack


Python
librosa — audio loading, visualization, and feature extraction
scikit-learn — Random Forest, Logistic Regression, Naive Bayes, Decision Tree, evaluation metrics
TensorFlow / Keras — custom TCN with Attention model
pandas, NumPy — data handling
matplotlib, seaborn — visualizations (confusion matrices, training curves, radar plots)
Google Colab — development environment


How to Run


Open Guardian_Deity_Women_Safety_Updated.ipynb in Google Colab
Mount your Google Drive and update the dataset path to point to your copy of the audio dataset
Run the cells sequentially — the notebook covers data loading, feature extraction, model training, and evaluation in order


Future Direction

As described in the project motivation, the long-term application of this work is integration into a wearable/IoT-based emergency alert system (pendant, earrings, rings, or mobile app) that could use audio-based danger detection as an automatic trigger alongside manual emergency buttons, sending real-time alerts to police or emergency contacts.
