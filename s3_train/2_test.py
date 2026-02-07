import s0_utils.global_params as g
from s0_utils.Classifier import Classifier
from s0_utils.Logger import Logger
from tqdm import tqdm
from pathlib import Path

TEST_FILE = Path("s3_train/2_test.log")

class Result():
    def __init__(self, correct_label, song_results):
        self.correct_label = correct_label
        self.song_results = song_results

        self.result_count = len(song_results)
        self.top1_hits = sum(r[0] for r in song_results)
        self.top3_hits = sum(r[1] for r in song_results)

        self.top_1_rate = self.top1_hits / self.result_count
        self.top_3_rate = self.top3_hits / self.result_count

        # 2/3 correct is enough
        self.is_top_1 = (3 * self.top1_hits) >= (2 * self.result_count)
        self.is_top_3 = (3 * self.top3_hits) >= (2 * self.result_count)
    
    def to_str(self):
        return f"[{self.correct_label}] top: {self.is_top_1} ({self.top1_hits}/{self.result_count})" \
            + f" | top 3: {self.is_top_3} ({self.top3_hits}/{self.result_count})"

classifier = Classifier(g.NAME)

logger = Logger(TEST_FILE)

def test_playlist(playlist_dir):
    song_results = []
    test_songs = list(playlist_dir.glob("*.mp3"))

    for test_song in test_songs:
        top, _ = classifier.infer(test_song)
        if top is None or len(top) < 3:
            continue

        is_top_1 = top[0][0] == playlist_dir.name
        is_top_3 = is_top_1 or \
            (top[1][0] == playlist_dir.name) or \
            (top[2][0] == playlist_dir.name)

        song_results.append((int(is_top_1), int(is_top_3)))

    if len(song_results) == 0:
        return None

    return Result(playlist_dir.name, song_results)

results = []
# results.append(test_playlist(Path("test/Bossa Nova")))
playlist_dirs = list(g.TEST_DIR.iterdir())
for playlist_dir in tqdm(playlist_dirs, total=len(playlist_dirs)):
    if not playlist_dir.is_dir():
        continue

    result = test_playlist(playlist_dir)
    if result is None:
        continue

    results.append(result)

results.sort(key=lambda r: (r.is_top_1, r.is_top_3))

print("\n\n")
for r in results:
    logger.writeln(r.to_str())

result_count = len(results)
top_1_pass_count = sum(1 for r in results if r.is_top_1)
top_1_fail_count = result_count - top_1_pass_count
top_1_perc = round(top_1_pass_count/(top_1_pass_count+top_1_fail_count)*100, 4)
top_3_pass_count = sum(1 for r in results if r.is_top_3)
top_3_fail_count = result_count - top_3_pass_count
top_3_perc = round(top_3_pass_count/(top_3_pass_count+top_3_fail_count)*100, 4)

logger.writeln(f"\n[Top 1] Pass: ({top_1_pass_count}/{result_count}) ({top_1_perc}%) | Fail: ({top_1_fail_count}/{result_count})")
logger.writeln(f"[Top 3] Pass: ({top_3_pass_count}/{result_count}) ({top_3_perc}%) | Fail: ({top_3_fail_count}/{result_count})")
