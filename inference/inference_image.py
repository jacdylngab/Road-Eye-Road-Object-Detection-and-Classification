import torch
import numpy as np
from PIL import Image
import io
import cv2
from transforms import val_transform_infer
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes, save_image
from .inference_utils import (
    device, post_proceesing_predictions, denormalize, 
    IMAGENET_MEAN, IMAGENET_STD, BDD100K_CLASSES, trained_model, draw_bbox
)


# =============================================================================
# Inference 
# =============================================================================

# Load the model
model = trained_model() 

def uploaded_image_to_tensor(file):
    # the Image is changed to a tensor because the model excepts a tensor

    # Read uploaded file directly as PIL image
    pil_image = Image.open(file).convert("RGB")

    # Convert PIL -> numpy for Albumentations
    image_np = np.array(pil_image)

    # Apply transform
    #unsqueeze(0) → adds the batch dimension (FCOS expects (B, 3, H, W)). B = 1
    #.to(device) → moves the tensor to GPU/CPU depending on what your model is using
    transformed = val_transform_infer(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    return image_tensor

def inference_single_image(image_tensor):


    with torch.no_grad():
        # Make predictions
        outputs = model(image_tensor)

        preds = post_proceesing_predictions(model=model, outputs=outputs)[0] # single image

        # Draw bounding boxes
        image = denormalize(image_tensor[0], mean=np.array(IMAGENET_MEAN), std=np.array(IMAGENET_STD))
        #img_uint8 = (image * 255).to(torch.uint8)

        """
        boxed_image = draw_bounding_boxes(
            image=img_uint8,
            boxes=preds["boxes"],
            labels=[f"{BDD100K_CLASSES.get(l.item(), 'unknown')}: {s:.2f}" for l, s in zip(preds["labels"], preds["scores"])],
            colors="red",
            width=2
        )

        boxed_image = F.resize(boxed_image, size=[720, 1280])
        """

        boxed_image = draw_bbox(
                image=image,
                boxes=preds["boxes"],
                labels=preds["labels"],
                scores=preds["scores"])

        boxed_image = cv2.resize(boxed_image, (1280, 720))

        #final_image = boxed_image.float() / 255.0 

    return boxed_image

def tensor_to_png_buffer(boxed_image):
    # Change the image with bbox to a numpy array
    #boxed_np = boxed_image.permute(1, 2, 0).cpu().numpy() # (H, W, C)
    boxed_np = cv2.cvtColor(boxed_image, cv2.COLOR_RGB2BGR) 

    # Convert result back to PIL for sending
    img = Image.fromarray(boxed_np)

    buf = io.BytesIO() # an in-memory buffer (so you don't have to save the image to disk)
    img.save(buf, format="PNG") # write the image into the buffer
    buf.seek(0) # rewind to the start

    return buf


        
