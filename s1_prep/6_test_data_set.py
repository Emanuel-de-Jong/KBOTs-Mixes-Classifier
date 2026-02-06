import shutil
import s0_utils.global_params as g

if not g.TEST_DIR.exists():
    for subdir in g.TRAIN_PLAYLISTS_DIR.iterdir():
        if subdir.is_dir():
            mp3_files = sorted(subdir.glob("*.mp3"))
            if mp3_files:
                test_subdir = g.TEST_DIR / subdir.name
                test_subdir.mkdir(parents=True, exist_ok=True)

                src_file = mp3_files[0]
                dest_file = test_subdir / src_file.name
                shutil.move(str(src_file), str(dest_file))
