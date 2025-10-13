import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

X = np.load("X_emb.npy")
y = pd.read_csv("y_labels.csv").iloc[:, 0].astype(str)
le = LabelEncoder().fit(y)
Y = le.transform(y)

all_labels = np.unique(Y)

knn = KNeighborsClassifier(n_neighbors=3, metric="cosine", n_jobs=-1)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(knn, X, Y, cv=cv, scoring="accuracy")
print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

top3_scores = []
for train_idx, val_idx in cv.split(X, Y):
    knn.fit(X[train_idx], Y[train_idx])
    probs = knn.predict_proba(X[val_idx])
    top3 = top_k_accuracy_score(Y[val_idx], probs, k=3, labels=all_labels)
    top3_scores.append(top3)

print(f"Mean Top-3 accuracy: {np.mean(top3_scores):.3f}")

knn.fit(X, Y)

joblib.dump(knn, "playlist_knn.joblib")
joblib.dump(le, "label_encoder.joblib")
