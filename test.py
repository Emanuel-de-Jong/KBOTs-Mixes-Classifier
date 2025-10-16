import joblib
from pathlib import Path
from Classifier import Classifier

LOW_SONG_COUNT_TRES = 15

classifier = Classifier()
train_dir = Path("train")
test_dir = Path("test")

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

        train_dir_songs = list((train_dir / playlist_dir.name).glob("*.mp3"))
        low_song_count = len(train_dir_songs) <= LOW_SONG_COUNT_TRES

        last_song = songs[-1]
        top = classifier.infer(last_song)
        if top is None or len(top) == 0:
            continue

        first_result = top[0]

        passed = first_result[0] == playlist_dir.name
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
            "predicted": first_result[0],
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
