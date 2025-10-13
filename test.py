import joblib
from pathlib import Path
from Mert import Mert

mert = Mert()
music_dir = Path("music")
test_dir = Path("test")
cache_dir = Path("cache")
knn = joblib.load(cache_dir / "playlist_knn.joblib")
le  = joblib.load(cache_dir / "label_encoder.joblib")

results = []
pass_count = 0
fail_count = 0
filtered_pass_count = 0
filtered_fail_count = 0

for playlist_dir in test_dir.iterdir():
    if playlist_dir.is_dir():
        songs = sorted(playlist_dir.glob("*.mp3"))
        if not songs:
            continue

        music_dir_songs = list((music_dir / playlist_dir.name).glob("*.mp3"))
        low_song_count = len(music_dir_songs) < 10

        last_song = songs[-1]
        vec = mert.run(str(last_song))
        if vec is None:
            continue

        vec = vec.reshape(1, -1)
        probs = knn.predict_proba(vec)[0]
        pred_idx = probs.argmax()
        pred_label = le.classes_[pred_idx]

        passed = pred_label == playlist_dir.name
        if passed:
            pass_count += 1
            if not low_song_count:
                filtered_pass_count += 1
        else:
            fail_count += 1
            if not low_song_count:
                filtered_fail_count += 1

        results.append({
            "playlist": playlist_dir.name,
            "song": last_song.name,
            "predicted": pred_label,
            "passed": passed,
            "low_song_count": low_song_count
        })

results.sort(key=lambda x: not x["passed"])

def write(f, msg):
    f.write(f"{msg}\n")
    print(msg)

print("\n\n")
with open("test.log", "w", encoding="utf-8") as f:
    for r in results:
        write(f, f"{r['playlist']}: {r['song']} -> {r['predicted']} | Passed: {r['passed']}")

    write(f, f"\nPass: {pass_count}, Fail: {fail_count}, Pass%: {pass_count/(pass_count+fail_count)*100}")
    write(f, f"[FILTERED] Pass: {filtered_pass_count}, Fail: {filtered_fail_count}" + \
          f", Pass%: {filtered_pass_count/(filtered_pass_count+filtered_fail_count)*100}")
