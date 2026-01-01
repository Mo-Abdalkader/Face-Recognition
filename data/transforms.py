"""
Image Transformation Pipelines
Config-driven augmentation
"""

from torchvision import transforms


class DataTransforms:
    """Factory class for creating transform pipelines"""

    @staticmethod
    def get_train_transforms(config):
        """
        Training transforms with augmentation

        Args:
            config: Config object with augmentation parameters
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(
                p=config.AUGMENTATION['horizontal_flip_prob']
            ),
            transforms.RandomRotation(
                config.AUGMENTATION['rotation_degrees']
            ),
            transforms.ColorJitter(
                brightness=config.AUGMENTATION['brightness'],
                contrast=config.AUGMENTATION['contrast'],
                saturation=config.AUGMENTATION['saturation'],
                hue=config.AUGMENTATION['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])

    @staticmethod
    def get_val_transforms(config):
        """
        Validation transforms (no augmentation)

        Args:
            config: Config object
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])