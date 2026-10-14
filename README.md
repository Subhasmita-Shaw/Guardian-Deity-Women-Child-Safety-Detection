# Guardian Deity — Women & Child Safety Detection Using Audio Signals

An audio signal classification project that detects danger situations involving women and children. By analyzing audio features extracted from `.wav` files, multiple Machine Learning and Deep Learning models are trained and compared to classify sounds into three categories: **Women** (distress sounds from women), **Child** (distress sounds from children), and **Normal** (everyday ambient sounds).

This project was developed as part of a research paper on real-time danger detection systems.

## Project Motivation

The broader vision behind this project is a safety system where, in a danger situation, a person could trigger an alert through a wearable device (pendant, earring, ring) or mobile phone, sending an emergency signal to police or trusted contacts. This notebook focuses on the machine learning component of that vision: building and evaluating models that can automatically detect distress sounds from audio, which could serve as an automatic trigger for such an alert system in the future. The hardware/IoT integration is a future direction and is not part of the implementation in this notebook.

## Dataset & Feature Extraction

Dataset: **Audio Signal Dataset for Danger Detection of Women and Children**

Download: https://data.mendeley.com/datasets/gfvsdtnf3v (DOI 10.17632/gfvsdtnf3v.1)

The dataset has `.wav` files labeled into three classes (Women, Child, Normal) and comes already divided into a `train` folder and a separate `test` folder. For each audio file, these features are extracted with librosa:

| Feature | Description |
|---|---|
| ZCR (Zero Crossing Rate) | How often the signal crosses zero; captures noisiness |
| RMS (Root Mean Square Energy) | Average energy/loudness of the audio |
| Spectral Centroid | "Center of mass" of the spectrum; captures brightness |
| MFCC 1–13 | Mel Frequency Cepstral Coefficients; capture tonal/timbral characteristics |

Each clip is described by 16 numbers (averaged over the clip). Models are **trained on the `train` folder** (13,562 clips) and **tested on the separate `test` folder** (5,777 clips; 36 empty test files were skipped).

## Models Trained & Compared

- **Random Forest Classifier** (100 trees): ensemble of decision trees
- **Logistic Regression**: linear baseline model
- **Gaussian Naive Bayes**: probabilistic model based on Bayes' theorem
- **Decision Tree Classifier**: interpretable baseline
- **TCN (Temporal Convolutional Network) with Attention**: a deep learning model using dilated causal convolutions with residual connections and an attention layer, applied to the 16-feature vector of each clip

## Evaluation

Each model is evaluated using:

- **Confusion Matrix**: correct vs. incorrect predictions per class
- **Precision, Recall and F1-Score**: per-class performance
- **AUC (Area Under the ROC Curve)**: One-vs-Rest strategy for multi-class evaluation
- **5-fold cross-validation** for Random Forest on the training data

## Results (tested on the separate `test` folder)

| Model | Test accuracy |
|---|---|
| TCN with Attention | 85.5% |
| Random Forest | 78.8% |
| Logistic Regression | 77.4% |
| Decision Tree | 75.5% |
| Gaussian Naive Bayes | 75.0% |

## Key Takeaways

- TCN with Attention reached the highest test accuracy, but its results vary between training runs, and its input is a set of clip-level averages, not a true time sequence. Treat it as an experiment, not a firm ranking.
- The classical models (75–79%) are fast and easy to interpret, which suits real-time use.
- An earlier random 80/20 split of the training data gave about 97% for Random Forest. That number is too optimistic, because pieces of the same recording can appear in both parts. The separate `test` folder is the more reliable measure.
- The Women class is the hardest: for Random Forest, only about half of Women clips are detected (recall 0.50). For a safety application, missed danger calls matter most, so this is the main area to improve.

## Tech Stack

Python, librosa (audio features), scikit-learn (classical models and metrics), TensorFlow / Keras (TCN with Attention), pandas, NumPy, matplotlib, seaborn. Developed in Jupyter / VS Code.

## How to Run

1. Download the dataset zip (link above) and put it in the same folder as the notebook.
2. Install the libraries: `pip install librosa matplotlib numpy soundfile pandas scikit-learn seaborn openpyxl tensorflow`
3. Open `Guardian_Deity_Women_Safety_Updated.ipynb` and run all cells. Feature extraction takes several minutes.

## Future Direction

As described in the project motivation, the long-term application of this work is integration into a wearable/IoT-based emergency alert system (pendant, earrings, rings, or mobile app) that could use audio-based danger detection as an automatic trigger alongside manual emergency buttons, sending real-time alerts to police or emergency contacts.

