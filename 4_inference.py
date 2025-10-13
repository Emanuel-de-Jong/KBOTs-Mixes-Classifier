import joblib
import sys
from Mert import Mert

path = "music/Bassline/TS7, Slick Don - Real Raver.mp3"
if len(sys.argv) > 1:
    path = sys.argv[1]

mert = Mert()
knn = joblib.load("playlist_knn.joblib")
le  = joblib.load("label_encoder.joblib")

vec = mert.run(path).reshape(1, -1)

probs = knn.predict_proba(vec)[0]
top = probs.argsort()[::-1][:5]
for i in top:
    print(f"{le.classes_[i]}: {probs[i]:.3f}")
