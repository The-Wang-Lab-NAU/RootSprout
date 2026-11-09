import os
import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from skimage import io

# ================= 1. Path helper function =================
def get_base_path():
    """Get the directory of the executable or script (compatible with PyInstaller packaging)."""
    if getattr(sys, 'frozen', False):
        # When packaged, sys._MEIPASS points to the temporary extraction directory.
        return sys._MEIPASS
    else:
        # Development environment
        return os.path.dirname(os.path.abspath(__file__))


# ================= 2. YOLO Model Class =================
class YOLO:
    def __init__(self, model_path=None):
        # If no path is provided, automatically build a relative path.
        if model_path is None:
            # Assume the weights file is located in the 'model_data' folder at the same level as the code, named 'best.pt'.
            model_path = os.path.join(get_base_path(), "model_data", "best.pt")
            
        print(f"Loading model from path: {model_path}")
        
        # Check if the file exists to prevent a direct crash if it's not found.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model weights file: {model_path}\nPlease ensure 'best.pt' is placed in the 'model_data' folder at the same level as the code, or pass the correct path manually.")
            
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Model loaded successfully! Classes: {self.class_names}")

    def detect_image(self, image, region_coord=None, seed_num=1, crop=False, count=False):
        """
        Parameters:
        - image: Input image (supports PIL.Image or numpy array)
        - region_coord: List of coordinates for 6 regions, or pass False/None to skip region statistics
        - seed_num: Total number of seeds used to calculate region ratios
        - crop: Whether to save the cropped single-seed images
        - count: Whether to print the count of each class
        """
        # If the input is a PIL Image, convert it to cv2 format (numpy array, BGR).
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()

        # 1. Model inference
        results = self.model(img, verbose=False)[0]

        boxes = results.boxes
        if len(boxes) == 0:
            return (Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                    [], [0, 0, 0, 0, 0, 0])

        # xyxy is in standard [x1, y1, x2, y2] format (left, top, right, bottom).
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)   # 0 or 1

        # ============================================================
        # ★ Map colors directly by class ID (BGR)
        #   0: class_0  → germinate     → Blue (255, 0, 0)
        #   1: class_1  → not germinate → Green (0, 255, 0)
        # ============================================================
        GERMINATE_ID     = 0
        NOT_GERMINATE_ID = 1

        COLOR_GERMINATE     = (255, 0, 0)   # Blue
        COLOR_NOT_GERMINATE = (0, 255, 0)   # Green
        COLOR_FALLBACK      = (0, 0, 255)   # Red (fallback for unknown ID)

        # Human-readable names (for label text)
        DISPLAY_NAME = {
            GERMINATE_ID:     "germinate",
            NOT_GERMINATE_ID: "not germinate",
        }

        def resolve_color(class_id):
            if class_id == GERMINATE_ID:
                return COLOR_GERMINATE
            if class_id == NOT_GERMINATE_ID:
                return COLOR_NOT_GERMINATE
            return COLOR_FALLBACK

        def resolve_name(class_id):
            # Use the mapping if available, otherwise fall back to the original name in model.names.
            return DISPLAY_NAME.get(class_id, self.class_names.get(class_id, f"class_{class_id}"))

        # Drawing parameters (adaptive to image size)
        H, W = img.shape[:2]
        thickness      = max(int(max(H, W) // 500), 2)
        font           = cv2.FONT_HERSHEY_SIMPLEX
        font_scale     = max(0.6, max(H, W) / 1500.0)
        font_thickness = max(1, thickness // 2)

        # Initialize statistics and lists
        Region_one, Region_two, Region_three = [], [], []
        Region_four, Region_five, Region_six = [], [], []
        class_counts = {name: 0 for name in self.class_names.values()}

        # 2. Iterate through each detected bounding box
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            confidence = float(conf[i])
            class_id   = int(cls[i])
            raw_name   = self.class_names[class_id]     # 'class_0' / 'class_1'
            disp_name  = resolve_name(class_id)         # 'germinate' / 'not germinate'

            class_counts[raw_name] += 1
            color = resolve_color(class_id)

            # ---------- Cropping ----------
            if crop:
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                # Use human-readable names for filenames to facilitate manual verification.
                cv2.imwrite(
                    os.path.join(dir_save_path, f"crop_{i}_{disp_name.replace(' ', '_')}.jpg"),
                    crop_image
                )

            # ---------- Region statistics (only count germinated class) ----------
            if (class_id == GERMINATE_ID and region_coord
                    and isinstance(region_coord, (list, tuple, np.ndarray))
                    and len(region_coord) >= 6):
                centroid_x = int(x1 + (x2 - x1) / 2)
                centroid_y = int(y1 + (y2 - y1) / 2)
                regions = [Region_one, Region_two, Region_three,
                        Region_four, Region_five, Region_six]
                for k in range(6):
                    c = region_coord[k]
                    if c[0] < centroid_x < c[2] and c[1] < centroid_y < c[3]:
                        regions[k].append(1)
                        break

            # ---------- Draw bounding box ----------
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # # ---------- Draw label (class name + confidence, refer to the first code snippet) ----------
            # label = f"{disp_name} {confidence:.2f}"
            # (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            # gap = 4
            # if y1 - th - baseline - gap > 0:
            #     # Enough space above → draw outside the box
            #     cv2.rectangle(img,
            #                 (x1, y1 - th - baseline - gap),
            #                 (x1 + tw, y1),
            #                 color, -1)
            #     text_y = y1 - baseline - gap // 2
            # else:
            #     # Not enough space above → draw inside the top of the box
            #     cv2.rectangle(img,
            #                 (x1, y1),
            #                 (x1 + tw, y1 + th + baseline + gap),
            #                 color, -1)
            #     text_y = y1 + th + gap // 2
            # cv2.putText(img, label, (x1, text_y),
            #             font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Print statistics
        if count:
            for name, num in class_counts.items():
                if num > 0:
                    print(f"{name}: {num}")

        # Calculate region ratios
        if region_coord and isinstance(region_coord, (list, tuple, np.ndarray)):
            if seed_num == 0:
                seed_num = 1
            radio = [
                np.sum(Region_one) / seed_num,
                np.sum(Region_two) / seed_num,
                np.sum(Region_three) / seed_num,
                np.sum(Region_four) / seed_num,
                np.sum(Region_five) / seed_num,
                np.sum(Region_six) / seed_num
            ]
        else:
            radio = [0, 0, 0, 0, 0, 0]

        r_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Convert to the [top, left, bottom, right] format required by the tracking code.
        if len(xyxy) > 0:
            xyxy_compatible = xyxy[:, [1, 0, 3, 2]]
        else:
            xyxy_compatible = []

        return r_image, xyxy_compatible, radio


# ================= 3. Main function and usage example =================
def DeepLearning_Models():
    """Load the YOLO model."""
    yolo = YOLO() 
    return yolo #, unet

if __name__ == "__main__":
    # 1. Load the model
    yolo = DeepLearning_Models()
    
    # 2. Read the image
    img_path = r"Time-series data_sorted\root_100.jpg"
    if not os.path.exists(img_path):
         raise FileNotFoundError(f"Cannot find test image: {img_path}")
         
    image = io.imread(img_path)
    
    # 3. Perform inference
    print("Starting inference...")
    r_image, BOX, radio = yolo.detect_image(
        Image.fromarray(image), 
        region_coord=False,  
        seed_num=9, 
        crop=False, 
        count=True
    )  
    
    # 4. Visualize the results
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)        
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(r_image)   
    plt.title('Detect Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nBounding box coordinates (BOX, converted to compatible format):")
    print(BOX)
    print("\nRegion ratios (Radio):", radio)