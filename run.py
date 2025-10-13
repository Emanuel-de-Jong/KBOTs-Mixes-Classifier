import joblib
import sys
from pathlib import Path
from Mert import Mert

path = "music/Bassline/TS7, Slick Don - Real Raver.mp3"
if len(sys.argv) > 1:
    path = sys.argv[1]

mert = Mert()
cache_dir = Path("cache")
knn = joblib.load(cache_dir / "playlist_knn.joblib")
le  = joblib.load(cache_dir / "label_encoder.joblib")

vec = mert.run(path).reshape(1, -1)

probs = knn.predict_proba(vec)[0]
top = probs.argsort()[::-1][:5]
for i in top:
    print(f"{le.classes_[i]}: {probs[i]:.3f}")
