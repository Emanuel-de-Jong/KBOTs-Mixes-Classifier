import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pathlib import Path

train_dir = Path("train")
train_dir.mkdir(exist_ok=True)
cache_dir = Path("cache")
X = np.load(cache_dir / "X_emb.npy")
y = pd.read_csv(cache_dir / "y_labels.csv")["labels"].astype(int)
labels = np.unique(pd.read_json(cache_dir / "num_to_label.json"))

cv = 4

test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=1)

def write(msg):
    with open(train_dir / "train.log", "a") as f:
        f.write(f"{msg}\n")
    print(msg)

train_distribution = y_train.value_counts()
for label_num, count in train_distribution.items():
    print(f"{labels[label_num]}: {count}")

model = KNeighborsClassifier(
    n_jobs=-1,
    metric='cosine',
    n_neighbors=3,
    weights='distance')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, target_names = labels)
write(report)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels = labels)
disp.plot()

plt.xticks(rotation=90)
plt.savefig(train_dir / f'Model.png')

joblib.dump(model, cache_dir / "model.joblib")
