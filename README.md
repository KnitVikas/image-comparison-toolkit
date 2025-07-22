# Image Comparison Toolkit

This script provides a comprehensive toolkit for comparing the similarity between two images using a variety of deep learning and classical computer vision techniques. It is designed to be a flexible and easy-to-use tool for developers and researchers working on tasks related to image analysis, object tracking, and content-based image retrieval.

The core of the toolkit is the `ImageComparator` class, which lazily loads models to ensure efficient memory usage.

## 🚀 Features & Comparison Methods

The script implements nine distinct methods for image comparison:

### Deep Learning Methods
1.  **CLIP (ViT-B/32):** Compares images based on semantic understanding. High scores mean the images are contextually similar, even if they look different.
2.  **DINOv2 (vit_base_patch14):** Uses self-supervised visual features to understand and compare object parts, textures, and shapes.
3.  **Deep SORT:** Leverages a re-identification model to generate robust descriptors for determining if two images contain the same object, even with changes in pose or lighting.

### Classical Computer Vision Methods
4.  **OpenCV Template Matching:** Finds a smaller image (template) within a larger one, robust to changes in brightness.
5.  **Pixel-level Change:** Calculates the percentage of differing pixels, offering a direct, low-level comparison.
6.  **Structural Similarity Index (SSIM):** A perceptual metric that compares images based on luminance, contrast, and structure, closely mimicking human perception.
7.  **Feature Matching (ORB):** Detects and matches hundreds of unique keypoints (like corners and blobs) between images.
8.  **Perceptual Hashing (pHash):** Creates a compact "fingerprint" of each image's visual structure and compares them.
9.  **Histogram Comparison:** Compares the color distribution of two images.

## 📦 Requirements & Installation

Before running the script, ensure you have Python installed. You will also need to install several third-party libraries. You can install them using pip:

```bash
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install timm
pip install numpy
pip install opencv-python
pip install scikit-image
pip install imagehash
pip install deep_sort_pytorch
```

You will also need a model checkpoint file for Deep SORT (`ckpt.t7`). You can typically find this within repositories that use Deep SORT.

##  usage

To use the script, you need to configure the main execution block at the bottom of the `compare_features.py` file.

1.  **Open the script:** `compare_features.py`
2.  **Set the image paths:** Modify the `image1_path` and `image2_path` variables to point to the two images you want to compare.
3.  **Set the Deep SORT model path:** Update the `deepsort_ckpt_path` variable to the location of your `ckpt.t7` file.

```python
if __name__ == "__main__":
    # --- Configuration ---
    image1_path = "/path/to/your/first/image.jpg"
    image2_path = "/path/to/your/second/image.jpg"
    deepsort_ckpt_path = "/path/to/your/ckpt.t7"

    # --- Main Execution ---
    try:
        comparator = ImageComparator(deepsort_model_path=deepsort_ckpt_path)
        # ... the rest of the script follows
```

4.  **Run the script from your terminal:**

```bash
python compare_features.py
```

## 📊 Example Output

The script will print the similarity scores from all nine methods, along with the time taken for each comparison. The output will look something like this:

```
✅ Using device: cuda

========================================
🔎 Method 1: CLIP Comparison
⏳ Loading CLIP model (ViT-B/32)...
CLIP Similarity: 0.8531 (took 5.1234s)

========================================
🦕 Method 2: DINOv2 Comparison
⏳ Loading DINOv2 model (vit_base_patch14_dinov2)...
DINOv2 Similarity: 0.9124 (took 2.4567s)

========================================
🚶‍♂️ Method 3: Deep SORT Comparison
⏳ Loading Deep SORT model from /path/to/your/ckpt.t7...
Deep SORT Similarity: 0.9587 (took 0.1234s)

========================================
🖼️ Method 4: Template Matching Comparison
Template Matching Similarity: 0.7845 (took 0.0123s)

========================================
⚫ Method 5: Pixel Change Comparison
Pixel Change Percentage (Dissimilarity): 15.40% (took 0.0045s)

========================================
📊 Method 6: Structural Similarity (SSIM)
SSIM Score: 0.6543 (took 0.0056s)

========================================
🔑 Method 7: Feature Matching (ORB)
Good Feature Matches (Score): 250.0 (took 0.0345s)

========================================
##️⃣ Method 8: Perceptual Hashing (pHash)
Hashing Similarity: 0.9219 (took 0.0098s)

========================================
🎨 Method 9: Histogram Comparison
Histogram Correlation: 0.8912 (took 0.0032s)
```
