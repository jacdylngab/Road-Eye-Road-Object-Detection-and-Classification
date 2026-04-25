import json
from tqdm import tqdm
from pathlib import Path


LABELS_TRAIN = "BDD100K Dataset/bdd100k_labels/100k/train"
LABELS_VALIDATION = "BDD100K Dataset/bdd100k_labels/100k/val"
LABELS_TEST = "BDD100K Dataset/bdd100k_labels/100k/test"


bus            = 0
traffic_light  = 0
traffic_sign   = 0
person         = 0
bike           = 0
truck          = 0
motor          = 0
car            = 0
train          = 0
rider          = 0


def count(labels_path):
    global bus, traffic_light, traffic_sign, person, bike, truck, motor, car, train, rider
    
    with open(labels_path, "r") as f:
        label_data = json.load(f)

    # Extract objects from the frame
    objects = label_data["frames"][0]["objects"]

    for obj in objects:
        if "box2d" in obj:
            if obj["category"] == "bus":
                bus += 1
            elif obj["category"] == "traffic light":
                traffic_light += 1
            elif obj["category"] == "traffic sign":
                traffic_sign += 1
            elif obj["category"] == "person":
                person += 1
            elif obj["category"] == "bike":
                bike += 1
            elif obj["category"] == "truck":
                truck += 1
            elif obj["category"] == "motor":
                motor += 1
            elif obj["category"] == "car":
                car += 1
            elif obj["category"] == "train":
                train += 1
            elif obj["category"] == "rider":
                rider += 1
            else:
                print("Don't what that object is")

train_dir = Path(LABELS_TRAIN)
val_dir = Path(LABELS_VALIDATION)
test_dir = Path(LABELS_TEST)

pbar_train =  tqdm(train_dir.iterdir(), desc=f"Counting Training Labels")
for label in pbar_train:
    count(label)

pbar_val =  tqdm(val_dir.iterdir(), desc=f"Counting Validation Labels")
for label in pbar_val:
    count(label) 

pbar_test =  tqdm(test_dir.iterdir(), desc=f"Counting Testing Labels")
for label in pbar_test:
    count(label)


with open("class_count.txt", "w") as file:
    file.write(f"bus = {bus}\n")
    file.write(f"traffic light = {traffic_light}\n")
    file.write(f"traffic_sign = {traffic_sign}\n")
    file.write(f"person = {person}\n")
    file.write(f"bike = {bike}\n")
    file.write(f"truck = {truck}\n")
    file.write(f"motor = {motor}\n")
    file.write(f"car = {car}\n")
    file.write(f"train = {train}\n")
    file.write(f"rider = {rider}\n")
    

print("DONE!")
