import global_params as g
from Classifier import Classifier
from pathlib import Path
from Utils import Logger
from tqdm import tqdm

class Result():
    def __init__(self, correct_label, top, song):
        self.correct_label = correct_label
        self.top = top
        self.song = song

        self.is_top_1 = top[0][0] == correct_label
        self.is_top_3 = self.is_top_1 or \
            (top[1][0] == correct_label) or \
            (top[2][0] == correct_label)
    
    def to_str(self):
        return f'[{self.correct_label}] top: {self.is_top_1} {"" if self.is_top_1 else f"({self.top[0][0]}) "}| top 3: {self.is_top_3}'

classifier = Classifier("global")

logger = Logger("test.log")

def test_playlist(playlist_dir):
    test_song = list(playlist_dir.glob("*.mp3"))[0]
    
    top, _ = classifier.infer(test_song)
    if top is None or len(top) == 0:
        return None

    return Result(playlist_dir.name, top, test_song.name)

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
top_1_perc = top_1_pass_count/(top_1_pass_count+top_1_fail_count)*100
top_3_pass_count = sum(1 for r in results if r.is_top_3)
top_3_fail_count = result_count - top_3_pass_count
top_3_perc = top_3_pass_count/(top_3_pass_count+top_3_fail_count)*100

logger.writeln(f"\n[Top 1] Pass: ({top_1_pass_count}/{result_count}) ({top_1_perc}%) | Fail: ({top_1_fail_count}/{result_count})")
logger.writeln(f"[Top 3] Pass: ({top_3_pass_count}/{result_count}) ({top_3_perc}%) | Fail: ({top_3_fail_count}/{result_count})")
