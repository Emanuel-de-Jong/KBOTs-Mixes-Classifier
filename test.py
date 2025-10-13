import joblib
import sys
from pathlib import Path
from Mert import Mert

path = "music/Bassline/TS7, Slick Don - Real Raver.mp3"

mert = Mert()
cache_dir = Path("cache")
knn = joblib.load(cache_dir / "playlist_knn.joblib")
le  = joblib.load(cache_dir / "label_encoder.joblib")

music_dir = Path("music")
results = []
pass_count = 0
fail_count = 0

for genre_dir in music_dir.iterdir():
    if genre_dir.is_dir():
        songs = sorted(genre_dir.glob("*.mp3"))
        if not songs:
            continue

        last_song = songs[-1]
        vec = mert.run(str(last_song))
        if vec is None:
            continue

        vec = vec.reshape(1, -1)
        probs = knn.predict_proba(vec)[0]
        pred_idx = probs.argmax()
        pred_label = le.classes_[pred_idx]

        passed = pred_label == genre_dir.name
        if passed:
            pass_count += 1
        else:
            fail_count += 1

        results.append({
            "genre": genre_dir.name,
            "song": last_song.name,
            "predicted": pred_label,
            "passed": passed
        })

results.sort(key=lambda x: not x["passed"])

def write(f, msg):
    f.write(f"{msg}\n")
    print(msg)

print("\n\n")
with open("test.log", "w", encoding="utf-8") as f:
    for r in results:
        write(f, f"{r['genre']}: {r['song']} -> {r['predicted']} | Passed: {r['passed']}")

    write(f, f"\nPass: {pass_count}, Fail: {fail_count}, Pass%: {pass_count/(pass_count+fail_count)*100}")
