import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# STEP 1: TRAINING PHASE
# ---------------------------

# Path to your training .wav files (each ~0.035s)
train_dir = 'Audio proccessing/DATA/Acoustic_train_test 2/EII_Raw_Ac_Fusion_Data_Train' 
train_files = [f for f in os.listdir(train_dir) if f.endswith('.wav')]
train_files.sort()

# Crate labels
def get_label(fname) -> str:
    if 'A2' in fname or 'drone' in fname:
        return 'drone'
    elif 'V1' in fname or 'car' in fname:
        return 'car'
    else:
        return 'unknown'

# Extract features from a .wav file
def extract_features_from_file(file_path, sr=48000):
    y, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    return np.hstack([mfccs, chroma, spec_cent, spec_bw, rms, zcr])


X_train = []
y_train = []

for fname in train_files:
    path = os.path.join(train_dir, fname)
    features = extract_features_from_file(path)
    X_train.append(features)
    y_train.append(get_label(fname))

X_train = np.array(X_train)


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

Xtr, Xval, ytr, yval = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(Xtr, ytr)

print(X_train)
print(y_train)
print("=== Validation report ===")
yval_pred = clf.predict(Xval)
print(classification_report(yval, yval_pred, target_names=le.classes_))

# Training Samples
test_dir = '/Users/nicolasgonzalez/Desktop/AFRL/Audio proccessing/DATA/Acoustic_train_test 2/EII_Raw_Ac_Fusion_Data_Test' 
# Just 
test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
test_files.sort()

# Same process to extract the features
X_test = []
for fname in test_files:
    path = os.path.join(test_dir, fname)
    features = extract_features_from_file(path)
    X_test.append(features)

X_test = np.array(X_test)

# Predict using trained model
y_pred = clf.predict(X_test)
predicted_labels = le.inverse_transform(y_pred)

# Show results
print("\n=== Prediction Results===")
for fname, label in zip(test_files, predicted_labels):
    print(f"{fname} -> {label}")
 