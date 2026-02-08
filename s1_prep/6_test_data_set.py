import shutil
import random
import s0_utils.global_params as g

TESTS_PER_GENRE = 3

if not g.TEST_DIR.exists():
    for subdir in g.TRAIN_PLAYLISTS_DIR.iterdir():
        if not subdir.is_dir():
            continue

        mp3_files = list(subdir.glob("*.mp3"))
        random.shuffle(mp3_files)
        if len(mp3_files) == 0:
            continue

        test_subdir = g.TEST_DIR / subdir.name
        test_subdir.mkdir(parents=True, exist_ok=True)

        tests_remaining = TESTS_PER_GENRE
        for i in range(TESTS_PER_GENRE):
            if len(mp3_files) < i + 1:
                break
            
            src_file = mp3_files[i]
            dest_file = test_subdir / src_file.name
            shutil.move(str(src_file), str(dest_file))

            tests_remaining -= 1
        
        if tests_remaining == 0:
            continue

        public_subdir = g.TRAIN_PUBLIC_PLAYLISTS_DIR / subdir.name
        if not public_subdir.exists():
            continue

        public_mp3_files = list(public_subdir.rglob("*.mp3"))
        random.shuffle(public_mp3_files)
        for i in range(tests_remaining):
            src_file = public_mp3_files[i]
            dest_file = test_subdir / src_file.name
            shutil.move(str(src_file), str(dest_file))
