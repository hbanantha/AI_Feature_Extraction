"""
Memory-Efficient Data Loader for Drone Image Feature Extraction

Lazy-loading dataset with on-the-fly augmentation for training
on systems with limited RAM.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================
# DroneImageDataset
# ==============================================================

class DroneImageDataset(Dataset):
    """
    Memory-efficient dataset for drone image tiles.
    Supports lazy loading and on-the-fly augmentation.
    """

    def __init__(
        self,
        tiles_dir: str,
        masks_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        is_training: bool = True,
        load_to_memory: bool = False,
        max_samples: Optional[int] = None,
        village_names: Optional[List[str]] = None,
        split_ratio: float = 0.8,
        split_seed: int = 42,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        self.is_training = is_training
        self.load_to_memory = load_to_memory
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.tile_paths: List[Path] = []
        self.mask_paths: List[Optional[Path]] = []

        self._collect_tiles(village_names, max_samples)
        self._apply_train_val_split()

        self.cached_data = None
        if load_to_memory:
            self._preload_to_memory()

        logger.info(f"Dataset initialized with {len(self.tile_paths)} tiles")

    def _collect_tiles(
        self,
        village_names: Optional[List[str]],
        max_samples: Optional[int],
    ):
        if village_names:
            villages_to_process = [
                self.tiles_dir / name
                for name in village_names
                if (self.tiles_dir / name).exists()
            ]
        else:
            potential_villages = [
                d for d in self.tiles_dir.iterdir() if d.is_dir()
            ]
            villages_to_process = (
                potential_villages if potential_villages else [self.tiles_dir]
            )
        logger.info(f"is_training = {self.is_training}")
        for village_dir in villages_to_process:
            tiles_subdir = (
                village_dir / "tiles"
                if (village_dir / "tiles").exists()
                else village_dir
            )

            for ext in [".npy", ".tif", ".tiff", ".png", ".jpg"]:
                for tile_path in tiles_subdir.glob(f"*{ext}"):

                    if max_samples and len(self.tile_paths) >= max_samples:
                        return

                    mask_path = None

                    if self.masks_dir:
                        # Build mask map only once (IMPORTANT optimization)
                        if not hasattr(self, "mask_map"):
                            self.mask_map = {
                                m.stem: m for m in self.masks_dir.rglob("*")
                            }

                        mask_path = self.mask_map.get(tile_path.stem)

                    # 🔥 LOAD MASK FOR FILTERING
                    keep = True

                    if mask_path is not None:
                        try:
                            if mask_path.suffix == ".npy":
                                mask = np.load(mask_path)
                            else:
                                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                            if np.any(mask > 0):
                                keep = True
                            else:
                                # Keep a small percentage of background tiles
                                keep = np.random.rand() < 0.15

                        except Exception:
                            logger.debug(f"Skipping corrupted mask: {mask_path}")
                            keep = False

                    # ✅ FINAL ADD
                    if keep:
                        self.tile_paths.append(tile_path)
                        self.mask_paths.append(mask_path)

            logger.info(f"Final tiles after filtering: {len(self.tile_paths)}")

    def _apply_train_val_split(self):
        """
        Apply train-validation split after filtering.
        Splitting is performed on tile_paths and mask_paths.
        Ensures no overlap between training and validation datasets.
        """
        if self.split_ratio is None:
            logger.info("Tile splitting disabled.")
            return

        if not (0.0 < self.split_ratio < 1.0):
            logger.warning(f"Invalid split_ratio: {self.split_ratio}")
            return

        total_tiles = len(self.tile_paths)

        if total_tiles == 0:
            logger.warning("No tiles available for splitting.")
            return

        import random
        random.seed(self.split_seed)

        indices = list(range(total_tiles))
        random.shuffle(indices)

        split_idx = int(total_tiles * self.split_ratio)

        if self.is_training:
            selected_indices = indices[:split_idx]
            split_type = "Training"
        else:
            selected_indices = indices[split_idx:]
            split_type = "Validation"

        # Apply split
        self.tile_paths = [self.tile_paths[i] for i in selected_indices]
        self.mask_paths = [self.mask_paths[i] for i in selected_indices]

        logger.info(
            f"{split_type} split applied: {len(self.tile_paths)} tiles "
            f"(split_ratio={self.split_ratio})"
        )

    def _find_mask_path(
        self, tile_path: Path, village_name: str
    ) -> Optional[Path]:

        tile_name = tile_path.stem

        possible_paths = [
            self.masks_dir / village_name / "masks" / f"{tile_name}.npy",
            self.masks_dir / village_name / "masks" / f"{tile_name}.png",
            self.masks_dir / village_name / f"{tile_name}.npy",
            self.masks_dir / village_name / f"{tile_name}.png",
            self.masks_dir / f"{tile_name}.npy",
            self.masks_dir / f"{tile_name}.png",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _preload_to_memory(self):
        logger.info("Preloading data to memory...")
        self.cached_data = []

        for idx in range(len(self.tile_paths)):
            tile = self._load_tile(self.tile_paths[idx])
            mask = (
                self._load_mask(self.mask_paths[idx])
                if idx < len(self.mask_paths)
                else None
            )
            self.cached_data.append((tile, mask))

        logger.info(f"Preloaded {len(self.cached_data)} samples")

    def _load_tile(self, path: Path) -> np.ndarray:
        if path.suffix == ".npy":
            return np.load(path)

        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None or not path.exists():
            return None

        if path.suffix == ".npy":
            return np.load(path)

        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cached_data is not None:
            tile, mask = self.cached_data[idx]
        else:
            tile = self._load_tile(self.tile_paths[idx])
            mask = None
            if idx < len(self.mask_paths):
                mask = self._load_mask(self.mask_paths[idx])

        if mask is None:
            mask = np.zeros((tile.shape[0], tile.shape[1]), dtype=np.int64)

            # Ensure mask matches tile dimensions
        if mask.shape[:2] != tile.shape[:2]:
            mask = cv2.resize(mask, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=tile, mask=mask)
            tile = transformed["image"]
            mask = transformed["mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask = mask.long()
        else:
            tile = torch.from_numpy(tile.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {"image": tile, "mask": mask, "path": str(self.tile_paths[idx])}

        # =====================================

        # if self.cached_data is not None:
        #     tile, mask = self.cached_data[idx]
        # else:
        #     tile = self._load_tile(self.tile_paths[idx])
        #     mask = None
        #     if idx < len(self.mask_paths):
        #         mask = self._load_mask(self.mask_paths[idx])
        #
        # if mask is None:
        #     mask = np.zeros(
        #         (tile.shape[0], tile.shape[1]), dtype=np.int64
        #     )
        #
        # if self.transform:
        #     transformed = self.transform(image=tile, mask=mask)
        #     tile = transformed["image"]
        #     mask = transformed["mask"]
        #     if not isinstance(mask, torch.Tensor):
        #         mask = torch.from_numpy(mask)
        #
        #     if mask.ndim == 3:
        #         mask = mask.squeeze(0)
        #
        #     mask = mask.long()
        # else:
        #     tile = (
        #         torch.from_numpy(tile.transpose(2, 0, 1)).float() / 255.0
        #     )
        #     mask = torch.from_numpy(mask).long()
        #
        # return {
        #     "image": tile,
        #     "mask": mask,
        #     "path": str(self.tile_paths[idx]),
        # }


# ==============================================================
# IncrementalDataset
# ==============================================================

class IncrementalDataset(Dataset):
    """
    Wrapper around DroneImageDataset that adds replay buffer support
    for incremental/continual learning.
    """

    def __init__(
        self,
        base_dataset: DroneImageDataset,
        replay_buffer: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        replay_ratio: float = 0.2,
    ):
        """
        Args:
            base_dataset: The underlying DroneImageDataset
            replay_buffer: List of (image, mask) tuples from previous tasks
            replay_ratio: Ratio of replay samples to use (0.0 to 1.0)
        """
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer or []
        self.replay_ratio = replay_ratio

    def __len__(self) -> int:
        base_len = len(self.base_dataset)
        replay_len = int(len(self.replay_buffer) * self.replay_ratio)
        return base_len + replay_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_len = len(self.base_dataset)
        if idx < base_len:
            return self.base_dataset[idx]
        else:
            replay_idx = idx - base_len
            replay_idx = replay_idx % len(self.replay_buffer)
            image, mask = self.replay_buffer[replay_idx]

            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            if self.base_dataset.transform:
                transformed = self.base_dataset.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask).long()

            return {"image": image, "mask": mask, "path": f"replay_buffer_{replay_idx}"}

    # base_len = len(self.base_dataset)
        #
        # if idx < base_len:
        #     # Return from base dataset
        #     return self.base_dataset[idx]
        # else:
        #     # Return from replay buffer
        #     replay_idx = idx - base_len
        #     replay_idx = replay_idx % len(self.replay_buffer)
        #
        #     image, mask = self.replay_buffer[replay_idx]
        #
        #     # Apply same transforms as base dataset if available
        #     if self.base_dataset.transform:
        #         transformed = self.base_dataset.transform(
        #             image=image, mask=mask
        #         )
        #         image = transformed["image"]
        #         mask = transformed["mask"]
        #     else:
        #         image = (
        #             torch.from_numpy(image.transpose(2, 0, 1)).float()
        #             / 255.0
        #         )
        #         mask = torch.from_numpy(mask).long()
        #
        #     return {
        #         "image": image,
        #         "mask": mask,
        #         "path": f"replay_buffer_{replay_idx}",
        #     }


# ==============================================================
# ReplayBuffer
# ==============================================================

class ReplayBuffer:
    """Replay buffer for incremental learning."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: List[Tuple[np.ndarray, np.ndarray]] = []

    def add(self, images: np.ndarray, masks: np.ndarray):
        for i in range(len(images)):

            if len(self.buffer) >= self.max_size:
                idx = np.random.randint(0, self.max_size)
                self.buffer[idx] = (
                    images[i].copy(),
                    masks[i].copy(),
                )
            else:
                self.buffer.append(
                    (images[i].copy(), masks[i].copy())
                )

    def sample(self, n: int):
        n = min(n, len(self.buffer))
        indices = np.random.choice(
            len(self.buffer), size=n, replace=False
        )
        return [self.buffer[i] for i in indices]

    def get_all(self):
        return self.buffer.copy()

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        np.save(path, np.array(self.buffer, dtype=object))

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.buffer = list(data)




# ==============================================================
# Augmentation Functions
# ==============================================================

# def get_training_augmentation(config: Dict) -> A.Compose:
#     """
#     Create training augmentation pipeline.
#
#     Args:
#         config: Configuration dictionary
#
#     Returns:
#         Albumentations Compose object
#     """
#     aug_config = config.get("augmentation", {}).get("train", {})
#
#     return A.Compose(
#         [
#             A.HorizontalFlip(p=aug_config.get("horizontal_flip_p", 0.5)),
#             A.VerticalFlip(p=aug_config.get("vertical_flip_p", 0.5)),
#             A.Rotate(
#                 limit=aug_config.get("rotate_limit", 45),
#                 p=aug_config.get("rotate_p", 0.5),
#             ),
#             A.GaussNoise(p=aug_config.get("gauss_noise_p", 0.2)),
#             A.OneOf(
#                 [
#                     A.GaussianBlur(),
#                     A.MotionBlur(),
#                 ],
#                 p=aug_config.get("blur_p", 0.2),
#             ),
#             A.OneOf(
#                 [
#                     A.OpticalDistortion(),
#                     A.GridDistortion(),
#                 ],
#                 p=aug_config.get("distortion_p", 0.1),
#             ),
#             A.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#             ),
#             ToTensorV2(),
#         ],
#         keypoint_params=None,
#     )
def get_training_augmentation(config: Dict) -> A.Compose:
    """
    Advanced augmentation pipeline for satellite imagery.
    
    Includes transformations that preserve structure while adding robustness:
    - Geometric: Flips, rotations, perspective shifts
    - Radiometric: Brightness, contrast, color shifts (simulate different sensors)
    - Noise: Gaussian noise, blur (simulate atmospheric effects)
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        
        # Perspective and elastic deformations (light)
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        
        # Radiometric augmentations (satellite-specific)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),  # Contrast Limited Adaptive Histogram Equalization
        
        # Color channel manipulations (simulate different bands/sensors)
        A.RandomRain(p=0.1),  # Simulate atmospheric effects
        A.GaussNoise(p=0.2),
        
        # Blur and edge-preserving filtering
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ],
    is_check_shapes=False,  # For replay buffer compatibility
    )

def get_validation_augmentation(config: Dict) -> A.Compose:
    """
    Create validation augmentation pipeline (minimal).
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        is_check_shapes=False,  # Disable strict shape checking
        keypoint_params=None,
    )


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get augmentation pipelines
    train_transform = get_training_augmentation(config)
    val_transform = get_validation_augmentation(config)
    
    # Create datasets
    train_dataset = DroneImageDataset(
        tiles_dir=config["data"]["tiles_dir"],
        masks_dir=config["data"].get("annotations_dir"),
        transform=train_transform,
        is_training=True,
        load_to_memory=False #True for GPU
    )
    
    val_dataset = DroneImageDataset(
        tiles_dir=config["data"]["tiles_dir"],
        masks_dir=config["data"].get("annotations_dir"),
        transform=val_transform,
        is_training=False,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 4),
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=config["hardware"].get("pin_memory", False),
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=config["hardware"].get("pin_memory", False),
    )
    
    return train_loader, val_loader


# ==============================================================
# Standalone Test Block
# ==============================================================

if __name__ == "__main__":

    print("Testing Data Loader Module...")

    config_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "config.yaml"
    )

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
    else:
        config = {
            "data": {
                "tiles_dir": "data/tiles",
                "annotations_dir": "data/annotations",
            },
            "training": {
                "batch_size": 2,
                "num_workers": 2,
            },
            "hardware": {
                "pin_memory": False,
            },
            "augmentation": {},
        }
        print("Using default fallback configuration")

    print("Data loader module initialized successfully!")