from pathlib import Path
import shutil

SHORTLIST_FOLDER_PATH = Path("short list") # Folder to store the inference pictures.
# Create the folder
SHORTLIST_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

INFERENCE_FOLDER_PATH = Path("inference_imgs")

IMAGES_TEST = Path("BDD100K Dataset/bdd100k_images_100k/100k/test")

TEST_PATH = Path("TEST")

TEST_SHORTLIST = Path("Test Short List")
TEST_SHORTLIST.mkdir(parents=True, exist_ok=True)

VIDEOS_PATH = Path("bdd100k/videos/test")

"""
nbr = [69,71,86,112,120,135,165,195,197,229,267,325,366,519,615,618,1095,1107,1384,1795,2426,2528,19652,18387,18306,18074,18001,17419,16547,16419,15719,12941,12140,12123,11875,9604,8932]

for image in TEST_PATH.iterdir():
    for n in nbr:
        if image.stem == f"image_{n}":
            src = TEST_PATH / f"{image.stem}.png"
            dst = TEST_SHORTLIST / f"{image.stem}.png"

            shutil.copy(src, dst)
"""
count = 1

for image in sorted(VIDEOS_PATH.iterdir()):
    old_file_path = image
    new_file_path = f"video_{count}.mov"
    count += 1

    old_file_path.rename(new_file_path)

print("DONE!")