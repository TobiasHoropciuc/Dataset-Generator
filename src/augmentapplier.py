from augmentations import Augmentations, BoxAugmentations
import configparser
import numpy as np

GENERAL_AUGMENTS = {
    "hsv_shift": Augmentations.hsv_shift,
    "gauss_noise": Augmentations.gauss_noise,
    "color_jitter": Augmentations.color_jitter,
}

CLASSIFICATION_AUGMENTS = {
    "resize": Augmentations.resize,
    "crop": Augmentations.crop,
    "flip": Augmentations.flip,
    "rotate": Augmentations.rotate,
    "random_erase": Augmentations.random_erase
}

DETECTION_AUGMENTS = {
    "resize": BoxAugmentations.resize,
    "flip": BoxAugmentations.flip,
    "random_object_erase": BoxAugmentations.random_object_erase
}

def apply(image: np.ndarray, augment_config: configparser.ConfigParser, boxes: list = None): # type: ignore
    for section in augment_config.sections():
        if section in GENERAL_AUGMENTS:
            image = GENERAL_AUGMENTS[section](image, dict(augment_config.items(section)))
        elif section in CLASSIFICATION_AUGMENTS and boxes is None:
            image = CLASSIFICATION_AUGMENTS[section](image, dict(augment_config.items(section)))
        elif section in DETECTION_AUGMENTS and boxes is not None:
            image, boxes = DETECTION_AUGMENTS[section](image, boxes, dict(augment_config.items(section)))
    return image, boxes
