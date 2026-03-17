import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training augmentations
train_transform = A.Compose([
    # 1. GEOMETRIC Transformation
    A.Resize( # Reduce the resolution of the images from 1280x720 to 640x360 for faster training
        height=360, 
        width=640
    ), 
    #A.Resize( # Reduce the resolution of the images from 1280x720 to 640x360 for faster training
    #    height=450, 
    #    width=800
    #), 
    A.HorizontalFlip(
        p=0.5
    ),
    A.Perspective(
        scale=(0.05, 0.1),
        keep_size=True,
        p=0.3
    ),

    # 2. COLOR / APPEARANCE Transformations
    A.RandomBrightnessContrast(
        brightness_limit=0.2, 
        contrast_limit=0.2, 
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10, 
        sat_shift_limit=20, 
        val_shift_limit=15, 
        p=0.3
    ),
    A.RandomShadow(
        p=0.2
    ),
    A.GaussianBlur(
        blur_limit=(3, 7),
        sigma_limit=(0.1, 2.0),
        p=0.2
    ),
    A.GaussNoise(
        std_range=(0.009, 0.015), 
        per_channel=True, 
        p=0.2
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
],
bbox_params=A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
    min_area=1,
    min_visibility=0.2
))

# Validation / Test augmentations
val_transform = A.Compose([
    A.Resize( # Reduce the resolution of the images from 1280x720 to 640x360 for faster training
        height=360, 
        width=640
    ), 
    #A.Resize( # Reduce the resolution of the images from 1280x720 to 640x360
    #    height=450, 
    #    width=800
    #), 
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
],
bbox_params=A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
    min_area=1,
    min_visibility=0.2
))

# Augmentation for inference on a single image.
# This does not except bounding boxes like other augmentations
val_transform_infer = A.Compose([
    A.Resize( # Reduce the resolution of the images from 1280x720 to 640x360 for faster training
        height=360, 
        width=640
    ), 
    #A.Resize( # Reduce the resolution of the images from 1280x720 to 640x360
    #    height=450, 
    #    width=800
    #), 
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])