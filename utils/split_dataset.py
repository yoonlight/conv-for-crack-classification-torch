"""
# Split dataset

## Reference

- [split dataset](https://stackoverflow.com/questions/17412439/how-to-split-data-into-trainset-and-testset-randomly)
- [](https://www.geeksforgeeks.org/python-list-files-in-a-directory/)
- [copy file](https://www.geeksforgeeks.org/python-shutil-copyfile-method/)

"""
import os
import random
import shutil
from pathlib import Path


def split_dataset(root_path, file_path):
    path = root_path / file_path

    TRAIN_PATH = root_path / "train" / file_path
    TEST_PATH = root_path / "test" / file_path
    # VAL_PATH = ROOT / "val"

    if os.path.exists(TRAIN_PATH) is False:
        os.makedirs(TRAIN_PATH)
    if os.path.exists(TEST_PATH) is False:
        os.makedirs(TEST_PATH)

    files = os.listdir(path)
    random.shuffle(files)

    train_data = files[:int((len(files)+1)*.80)]
    test_data = files[int((len(files)+1)*.80):]

    copy_files(train_data, path, TRAIN_PATH)
    copy_files(test_data, path, TEST_PATH)


def copy_files(files, src_path, dst_path):
    for file in files:
        copy_file(filename=file, src_path=src_path, dst_path=dst_path)


def copy_file(filename, src_path, dst_path):
    shutil.copy2(src_path / filename, dst_path / filename)


if __name__ == "__main__":
    ROOT = Path("./datasets/crack")
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    split_dataset(root_path=ROOT, file_path=POSITIVE)
    split_dataset(root_path=ROOT, file_path=NEGATIVE)
