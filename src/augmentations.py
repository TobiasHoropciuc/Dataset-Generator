import numpy
import cv2
import random
import numpy as np
import albumentations as A

class Augmentations:

    @staticmethod
    def resize(image: numpy.ndarray, args: dict) -> numpy.ndarray:
        return cv2.resize(image, (int(args['width']), int(args['height'])), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def random_erase(image: numpy.ndarray, args: dict) -> numpy.ndarray:
        if random.uniform(0, 1) <= float(args['prob']):
            max_width = image[0].__len__()
            max_height = image.__len__()
            crop_width = random.randint(0, int(max_width * float(args['max_erase'])))
            crop_height = random.randint(0, int(max_height * float(args['max_erase'])))
            crop_x0 = random.randint(0, max_width - crop_width)
            crop_y0 = random.randint(0, max_height - crop_height)
            crop_x1 = crop_x0 + crop_width
            crop_y1 = crop_y0 + crop_height
            for i in range(np.shape(image)[2]):
                image[crop_y0: crop_y1, crop_x0: crop_x1, i] = random.randint(0, 255)
        return image   


    @staticmethod
    def flip(image: numpy.ndarray, args: dict) -> numpy.ndarray:
        if random.uniform(0, 1) <= float(args['prob']):
            if int(args['type']) == 0:
                return cv2.flip(image, 0)
            else:
                return cv2.flip(image, 1)
        return image

    @staticmethod
    def crop(image: numpy.ndarray, args: dict) -> numpy.ndarray:
        if random.uniform(0, 1) <= float(args['prob']):
            width = image[0].__len__()
            height = image.__len__()
            factor = random.uniform(float(args['min_crop']), 1)
            crop_width = int(width * factor)
            crop_height = int(height * factor)
            x0 = random.randint(0, width - crop_width)
            y0 = random.randint(0, height - crop_height)
            crop = image[y0: (y0 + crop_height), x0: (x0 + crop_width)]
            image = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def rotate(image: np.ndarray, args: dict) -> np.ndarray:
        if random.uniform(0, 1) <= float(args['prob']):
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=random.uniform(float(args['min_degree']), float(args['max_degree'])), scale=1)
            image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        return image

    @staticmethod
    def color_jitter(image: np.ndarray, args: dict) -> np.ndarray:
        brighness = random.uniform(0, float(args['brightness']))
        contrast = random.uniform(0, float(args['contrast']))
        hue = random.uniform(0, float(args['hue']))
        saturation = random.uniform(0, float(args['saturation']))
        transforms = A.Compose([
            A.augmentations.transforms.ColorJitter(brightness=brighness,
                                                            contrast=contrast,
                                                            hue=hue,
                                                            saturation=saturation,
                                                            p=float(args['prob']))
        ])
        image = transforms(image=image)['image']
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image    
    
    @staticmethod
    def hsv_shift(image: np.ndarray, args: dict) -> np.ndarray:
        transforms = A.Compose([
            A.augmentations.transforms.HueSaturationValue(hue_shift_limit=int(args['hue']),
                                                            sat_shift_limit=int(args['saturation']),
                                                            val_shift_limit=int(args['value']),
                                                            p=float(args['prob']))
        ])
        image = transforms(image=image)['image']
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image    

    @staticmethod
    def gauss_noise(image: np.ndarray, args: dict) -> np.ndarray:
        print(image[100][100][0], image[100][100][1], image[100][100][2])
        transforms = A.Compose([
            A.GaussNoise(var_limit=int(args['variance']),
                        mean=int(args['mean']),
                        p=float(args['prob']))
        ])
        print(image[100][100][0], image[100][100][1], image[100][100][2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image[100][100][0], image[100][100][1], image[100][100][2])
        return transforms(image=image)['image']



'''
This class offers augmentations specific for the generation of object detection datasets.
The main difference to ClassificationAugmentations is that its augmentation methods adjust box coordinates too.
'''


class BoxAugmentations:

    @staticmethod
    def resize(image: np.ndarray, boxes: list, args: dict) -> tuple:
        aug_image = cv2.resize(image, (int(args['width']), int(args['height'])), interpolation=cv2.INTER_LINEAR)
        factor_x = int(args['width']) / image[0].__len__()
        factor_y = int(args['height']) / image.__len__()
        aug_boxes = []
        for box in boxes:
            aug_boxes.append({
                'x0': int(box['x0'] * factor_x),
                'y0': int(box['y0'] * factor_y),
                'x1': int(box['x1'] * factor_x),
                'y1': int(box['y1'] * factor_y)
            })
        return aug_image, aug_boxes

    @staticmethod #only used by overlay to vary object sizes
    def random_object_resize(image: np.ndarray, obj: np.ndarray) -> np.ndarray: 
        object_width = obj[0].__len__()
        object_height = obj.__len__()
        background_width = image[0].__len__()
        background_height = image.__len__()
        min_object_width = int(min([background_width, background_height]) / 12)
        max_object_width = int(min([background_width, background_height]) / 4)
        random_width = random.randint(min_object_width, max_object_width)
        factor = random_width / max([object_width, object_height])
        if object_width >= object_height:
            new_object_width = random_width
            new_object_height = object_height * factor
        else:
            new_object_height = random_width
            new_object_width = object_width * factor
        return cv2.resize(obj, (int(new_object_width), int(new_object_height)))

    @staticmethod
    def overlay_object(image: np.ndarray, obj: np.ndarray) -> tuple:
        obj = BoxAugmentations.random_object_resize(image, obj)
        max_width = image[0].__len__() - obj[0].__len__()
        max_height = image.__len__() - obj.__len__()
        start_width = random.randint(0, max_width)
        start_height = random.randint(0, max_height)
        for i in range(obj.__len__()):
            for j in range(obj[0].__len__()):
                if obj[i][j][3] != 0:
                    image[start_height + i][start_width + j][0] = obj[i][j][0]
                    image[start_height + i][start_width + j][1] = obj[i][j][1]
                    image[start_height + i][start_width + j][2] = obj[i][j][2]
        box = {
            'x0': start_width,
            'y0': start_height,
            'x1': start_width + obj[0].__len__(),
            'y1': start_height + obj.__len__()
        }
        return image, box

    @staticmethod
    def random_object_erase(image: np.ndarray, boxes: list, args: dict) -> tuple:
        for box in boxes:
            image[box['y0']:box['y1'], box['x0']:box['x1']] = Augmentations.random_erase(image[box['y0']:box['y1'],
                                                                                                        box['x0']:box['x1']], args['max_erase']) 
        return image, boxes

    @staticmethod
    def flip(image: np.ndarray, boxes: list, args:dict) -> tuple:
        if random.uniform(0, 1) <= float(args['prob']):
            flipped_boxes = []
            if int(args['type']) == 0:
                image = cv2.flip(image, 0)
                height = image.__len__()
                for box in boxes:
                    flipped_boxes.append({
                        'x0': box['x0'],
                        'y0': height - int(box['y1']),
                        'x1': box['x1'],
                        'y1': height - int(box['y0'])
                    }
                    )
                boxes = flipped_boxes
            else:
                image = cv2.flip(image, 1)
                width = image[0].__len__()
                for box in boxes:
                    flipped_boxes.append({
                        'x0': width - int(box['x1']),
                        'y0': box['y0'],
                        'x1': width - int(box['x0']),
                        'y1': box['y1']
                    })   
                boxes = flipped_boxes
        return image, boxes