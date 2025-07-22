import torch
import torch.nn.functional as F
from PIL import Image
import time
import clip
import timm
from torchvision import transforms
import os
import numpy as np
import cv2  # Added for template matching and pixel change

# You may need to install the 'deep_sort_pytorch', 'scikit-image', and 'imagehash' packages
# For example: pip install deep_sort_pytorch scikit-image imagehash
from deep_sort.deep.feature_extractor import Extractor
from skimage.metrics import structural_similarity as ssim
import imagehash


class ImageComparator:
    """
    A class to compare image similarity using different feature extraction models.

    This class provides methods to calculate the similarity between two images
    using nine different methods:
    1.  CLIP (ViT-B/32)
    2.  DINOv2 (vit_base_patch14)
    3.  Deep SORT's appearance feature extractor
    4.  OpenCV Template Matching
    5.  Pixel-level Change Calculation
    6.  Structural Similarity Index (SSIM)
    7.  Feature Matching (ORB)
    8.  Perceptual Hashing (pHash)
    9.  Histogram Comparison
    
    Models are loaded into memory on-demand to improve efficiency.
    """

    def __init__(self, deepsort_model_path: str):
        """
        Initializes the ImageComparator.

        Args:
            deepsort_model_path (str): The file path to the Deep SORT model checkpoint (.t7 file).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Using device: {self.device}")

        # --- Model placeholders for lazy loading ---
        self.clip_model = None
        self.clip_preprocess = None
        self.dino_model = None
        self.deepsort_extractor = None
        self.deepsort_model_path = deepsort_model_path

    # --------------------------------------------------------------------------
    # Private Model Loaders (Lazy Loading)
    # --------------------------------------------------------------------------

    def _load_clip(self):
        """Loads the CLIP model and preprocessor if not already loaded."""
        if self.clip_model is None:
            print("⏳ Loading CLIP model (ViT-B/32)...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()

    def _load_dino(self):
        """Loads the DINOv2 model if not already loaded."""
        if self.dino_model is None:
            print("⏳ Loading DINOv2 model (vit_base_patch14_dinov2)...")
            self.dino_model = timm.create_model("vit_base_patch14_dinov2", pretrained=True)
            self.dino_model.eval().to(self.device)

    def _load_deepsort(self):
        """Loads the Deep SORT feature extractor if not already loaded."""
        if self.deepsort_extractor is None:
            print(f"⏳ Loading Deep SORT model from {self.deepsort_model_path}...")
            if not os.path.exists(self.deepsort_model_path):
                raise FileNotFoundError(f"Deep SORT model not found at: {self.deepsort_model_path}")
            self.deepsort_extractor = Extractor(self.deepsort_model_path, use_cuda=(self.device == "cuda"))
            
    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------
    
    def _check_paths(self, image1_path: str, image2_path: str):
        """Checks if image paths exist."""
        if not os.path.exists(image1_path):
            raise FileNotFoundError(f"Image not found at: {image1_path}")
        if not os.path.exists(image2_path):
            raise FileNotFoundError(f"Image not found at: {image2_path}")

    # --------------------------------------------------------------------------
    # Public Comparison Methods
    # --------------------------------------------------------------------------

    def compare_with_clip(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Compares images using CLIP's semantic understanding.

        How it works:
        CLIP (Contrastive Language-Image Pre-Training) embeds images into a rich,
        multimodal space where semantically similar images are located close together.
        This method computes the feature vector (embedding) for each image and then
        calculates their cosine similarity. A high score indicates semantic closeness,
        even if the images are not visually identical.

        Score:
        - Type: Cosine Similarity
        - Range: [-1.0, 1.0]
        - Interpretation: Higher is more similar. 1.0 means identical embeddings.

        Resources:
        - https://openai.com/research/clip
        """
        self._check_paths(image1_path, image2_path)
        self._load_clip()
        start_time = time.time()

        image1 = self.clip_preprocess(Image.open(image1_path)).unsqueeze(0).to(self.device)
        image2 = self.clip_preprocess(Image.open(image2_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat1 = self.clip_model.encode_image(image1)
            feat2 = self.clip_model.encode_image(image2)

        similarity = F.cosine_similarity(feat1, feat2).item()
        return similarity, time.time() - start_time

    def compare_with_dino(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Compares images using DINOv2's self-supervised visual features.

        How it works:
        DINOv2 is a model trained without explicit labels, learning rich visual
        representations that are excellent at understanding object parts, textures,
        and shapes. Similar to CLIP, it encodes images into feature vectors, and
        their cosine similarity is computed.

        Score:
        - Type: Cosine Similarity
        - Range: [-1.0, 1.0]
        - Interpretation: Higher is more similar.

        Resources:
        - https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/
        """
        self._check_paths(image1_path, image2_path)
        self._load_dino()
        start_time = time.time()

        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        img1 = transform(Image.open(image1_path).convert("RGB")).unsqueeze(0).to(self.device)
        img2 = transform(Image.open(image2_path).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat1 = self.dino_model.forward_features(img1).mean(dim=1)
            feat2 = self.dino_model.forward_features(img2).mean(dim=1)

        similarity = F.cosine_similarity(feat1, feat2).item()
        return similarity, time.time() - start_time

    def compare_with_deepsort(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Compares images using Deep SORT's re-identification model.

        How it works:
        This method uses the appearance feature extractor from the Deep SORT object
        tracking algorithm. The model is specifically trained to generate descriptors
        that are robust to changes in pose and lighting, making it effective for
        determining if two images contain the same object.

        Score:
        - Type: Cosine Similarity
        - Range: [-1.0, 1.0]
        - Interpretation: Higher is more similar.

        Resources:
        - https://arxiv.org/abs/1703.07402
        """
        self._check_paths(image1_path, image2_path)
        self._load_deepsort()
        start_time = time.time()

        def load_image_np(path):
            img = Image.open(path).convert("RGB")
            return np.asarray(img)

        image1_np = load_image_np(image1_path)
        image2_np = load_image_np(image2_path)

        features = self.deepsort_extractor([image1_np, image2_np])
        feat1 = torch.tensor(features[0])
        feat2 = torch.tensor(features[1])

        similarity = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
        return similarity, time.time() - start_time

    def compare_with_template_matching(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Finds a smaller image (template) within a larger one (scene).

        How it works:
        This classical computer vision technique slides the smaller image (template)
        over every possible location of the larger image (scene). At each location,
        it calculates a correlation score. The method TM_CCOEFF_NORMED is used, which
        is robust to linear changes in brightness. The highest score represents the
        best match location.

        Score:
        - Type: Normalized Cross-Correlation
        - Range: [-1.0, 1.0] (clamped to [0.0, 1.0] in this implementation)
        - Interpretation: Higher is more similar.

        Resources:
        - https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
        """
        self._check_paths(image1_path, image2_path)
        start_time = time.time()

        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None: return 0.0, time.time() - start_time

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        template_img, scene_img = (img1, img2) if h1 * w1 < h2 * w2 else (img2, img1)
        
        try:
            result = cv2.matchTemplate(scene_img, template_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
        except cv2.error:
            return 0.0, time.time() - start_time

        similarity = float(max(0.0, max_val))
        return similarity, time.time() - start_time

    def compare_with_pixel_change(self, image1_path: str, image2_path: str, threshold: int = 30) -> tuple[float, float]:
        """
        Calculates the percentage of differing pixels between two images.

        How it works:
        This method performs a direct, low-level comparison. It resizes the images
        to be identical, converts them to grayscale, and computes the absolute
        difference for each corresponding pixel. A threshold is applied to create a
        binary mask of "changed" vs. "unchanged" pixels, and the percentage of
        changed pixels is calculated.

        Score:
        - Type: Dissimilarity Percentage
        - Range: [0.0, 100.0]
        - Interpretation: Higher means LESS similar (more different).

        Resources:
        - https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
        """
        self._check_paths(image1_path, image2_path)
        start_time = time.time()

        try:
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            if image1 is None or image2 is None: return 0.0, time.time() - start_time
        except Exception:
            return 0.0, time.time() - start_time

        target_dims = (image1.shape[1], image1.shape[0])
        image2_resized = cv2.resize(image2, target_dims, interpolation=cv2.INTER_AREA)

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        changed_pixels_count = cv2.countNonZero(thresh)
        total_pixels = image1.shape[0] * image1.shape[1]
        
        change_percentage = (changed_pixels_count / total_pixels) * 100 if total_pixels > 0 else 0.0
        return change_percentage, time.time() - start_time

    def compare_with_ssim(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Calculates the Structural Similarity Index (SSIM).

        How it works:
        SSIM is a perceptual metric that assesses image similarity based on three
        key components: luminance, contrast, and structure. It is designed to be
        closer to human perception of similarity than simple metrics like pixel
        difference or Mean Squared Error (MSE).

        Score:
        - Type: Structural Similarity
        - Range: [-1.0, 1.0]
        - Interpretation: Higher is more similar. 1.0 indicates a perfect match.

        Resources:
        - https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
        """
        self._check_paths(image1_path, image2_path)
        start_time = time.time()

        try:
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            if image1 is None or image2 is None: return 0.0, time.time() - start_time
        except Exception:
            return 0.0, time.time() - start_time

        target_dims = (image1.shape[1], image1.shape[0])
        image2_resized = cv2.resize(image2, target_dims, interpolation=cv2.INTER_AREA)

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

        similarity_score, _ = ssim(gray1, gray2, full=True)
        return similarity_score, time.time() - start_time

    def compare_with_feature_matching(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Compares images by finding and matching distinct keypoints (ORB).

        How it works:
        This method uses the ORB (Oriented FAST and Rotated BRIEF) algorithm to
        detect hundreds of unique keypoints (corners, blobs, etc.) in both images.
        It then computes a descriptor for each keypoint and uses a Brute-Force
        matcher to find corresponding pairs. Lowe's ratio test is applied to
        filter out weak or ambiguous matches, keeping only the "good" ones.

        Score:
        - Type: Raw count of good matches.
        - Range: [0, N] (where N is the number of features, e.g., 1000)
        - Interpretation: Higher number suggests more similarity. Not normalized.

        Resources:
        - https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
        """
        self._check_paths(image1_path, image2_path)
        start_time = time.time()

        try:
            img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None: return 0.0, time.time() - start_time

            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None: return 0.0, time.time() - start_time

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            if matches and len(matches) > 1 and len(matches[0]) == 2:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            score = float(len(good_matches))
            return score, time.time() - start_time
        except cv2.error:
            return 0.0, time.time() - start_time

    def compare_with_hashing(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Compares images using a perceptual hash (pHash).

        How it works:
        This method creates a compact "fingerprint" (hash) of each image's visual
        structure using the pHash algorithm. It then calculates the Hamming distance
        (the number of differing bits) between the two hashes. This distance is very
        small for visually similar images, even if they have been resized or have
        minor edits.

        Score:
        - Type: Normalized Similarity
        - Range: [0.0, 1.0]
        - Interpretation: Higher is more similar. 1.0 means identical hashes.

        Resources:
        - https://github.com/JohannesBuchner/imagehash
        """
        self._check_paths(image1_path, image2_path)
        start_time = time.time()

        try:
            hash1 = imagehash.phash(Image.open(image1_path))
            hash2 = imagehash.phash(Image.open(image2_path))
            distance = hash1 - hash2
            
            # Normalize the score. Lower distance is better.
            # Max distance for pHash is 64 bits.
            similarity = (64 - distance) / 64.0
            return similarity, time.time() - start_time
        except Exception:
            return 0.0, time.time() - start_time

    def compare_with_histogram(self, image1_path: str, image2_path: str) -> tuple[float, float]:
        """
        Compares the color distribution of two images via their histograms.

        How it works:
        This method computes a 3D color histogram for each image, which represents
        the amount of each color present. The histograms are normalized to make the
        comparison independent of image size. Finally, it calculates the correlation
        between the two histograms. A high correlation means the images have a
        similar color layout.

        Score:
        - Type: Histogram Correlation
        - Range: [-1.0, 1.0]
        - Interpretation: Higher is more similar. 1.0 means identical color distributions.

        Resources:
        - https://docs.opencv.org/4.x/d8/dc8/tutorial_py_histogram_comparison.html
        """
        self._check_paths(image1_path, image2_path)
        start_time = time.time()

        try:
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            if img1 is None or img2 is None: return 0.0, time.time() - start_time

            target_dims = (img1.shape[1], img1.shape[0])
            img2_resized = cv2.resize(img2, target_dims, interpolation=cv2.INTER_AREA)

            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist1 = cv2.normalize(hist1, hist1).flatten()

            hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2).flatten()

            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return score, time.time() - start_time
        except cv2.error:
            return 0.0, time.time() - start_time


if __name__ == "__main__":
    # --- Configuration ---
    image1_path = "/data-mount/yolov7/utils_vikas/changes_detect/frames/frame_12.jpg"
    image2_path = "/data-mount/yolov7/utils_vikas/changes_detect/frames/frame_19.jpg"
    deepsort_ckpt_path = "/data-mount/yolov7/utils_vikas/deep_sort_pytorch/ckpt.t7"

    # --- Main Execution ---
    try:
        comparator = ImageComparator(deepsort_model_path=deepsort_ckpt_path)

        # --- Deep Learning Methods ---
        print("\n" + "="*40)
        print("🔎 Method 1: CLIP Comparison")
        clip_sim, clip_time = comparator.compare_with_clip(image1_path, image2_path)
        print(f"CLIP Similarity: {clip_sim:.4f} (took {clip_time:.4f}s)")

        print("\n" + "="*40)
        print("🦕 Method 2: DINOv2 Comparison")
        dino_sim, dino_time = comparator.compare_with_dino(image1_path, image2_path)
        print(f"DINOv2 Similarity: {dino_sim:.4f} (took {dino_time:.4f}s)")
        
        print("\n" + "="*40)
        print("🚶‍♂️ Method 3: Deep SORT Comparison")
        deepsort_sim, deepsort_time = comparator.compare_with_deepsort(image1_path, image2_path)
        print(f"Deep SORT Similarity: {deepsort_sim:.4f} (took {deepsort_time:.4f}s)")
        
        # --- Classical Computer Vision Methods ---
        print("\n" + "="*40)
        print("🖼️ Method 4: Template Matching Comparison")
        template_sim, template_time = comparator.compare_with_template_matching(image1_path, image2_path)
        print(f"Template Matching Similarity: {template_sim:.4f} (took {template_time:.4f}s)")

        print("\n" + "="*40)
        print("⚫ Method 5: Pixel Change Comparison")
        pixel_change_pct, pixel_time = comparator.compare_with_pixel_change(image1_path, image2_path)
        print(f"Pixel Change Percentage (Dissimilarity): {pixel_change_pct:.2f}% (took {pixel_time:.4f}s)")

        print("\n" + "="*40)
        print("📊 Method 6: Structural Similarity (SSIM)")
        ssim_score, ssim_time = comparator.compare_with_ssim(image1_path, image2_path)
        print(f"SSIM Score: {ssim_score:.4f} (took {ssim_time:.4f}s)")

        print("\n" + "="*40)
        print("🔑 Method 7: Feature Matching (ORB)")
        feature_matches, feature_time = comparator.compare_with_feature_matching(image1_path, image2_path)
        print(f"Good Feature Matches (Score): {feature_matches} (took {feature_time:.4f}s)")

        print("\n" + "="*40)
        print("##️⃣ Method 8: Perceptual Hashing (pHash)")
        hash_sim, hash_time = comparator.compare_with_hashing(image1_path, image2_path)
        print(f"Hashing Similarity: {hash_sim:.4f} (took {hash_time:.4f}s)")

        print("\n" + "="*40)
        print("🎨 Method 9: Histogram Comparison")
        hist_sim, hist_time = comparator.compare_with_histogram(image1_path, image2_path)
        print(f"Histogram Correlation: {hist_sim:.4f} (took {hist_time:.4f}s)")

    except (FileNotFoundError, RuntimeError, ModuleNotFoundError) as e:
        print(f"\n❌ An error occurred: {e}")
        print("👉 Please ensure all file paths are correct and required libraries (timm, clip, deep_sort_pytorch, opencv-python, scikit-image, imagehash) are installed.")
