import os
import random
import types
from abc import ABC, abstractmethod
import cv2
from augmentations import BoxAugmentations
import augmentapplier
import annotationwriters
from generatorexception import InvalidGeneratorArgumentException
import configparser


class Generator(ABC):

    def __init__(self, image_source: str, img_output_dir: str, num: int, augment_config: configparser.ConfigParser):
        self.set_image_source(image_source)
        self.set_img_output_dir(img_output_dir)
        self.set_num(num)
        self.set_augment_config(augment_config)

    @staticmethod
    def valid_dir(dir: str) -> bool:
        if os.path.isdir(dir) and len(os.listdir(dir)) != 0:
            return True
        return False

    @staticmethod
    def dir_contains_only_imgs(dir: str) ->bool:
        for file in os.listdir(dir):
            img = cv2.imread(os.path.join(dir, file))
            # if img is of type NoneType then it cannot be a valid image file (expected: numpy.ndarray)
            if type(img) == types.NoneType:
                return False
        return True

    @staticmethod
    def get_random_file(dir: str) -> str:
        files = os.listdir(dir)
        return os.path.join(dir, files[random.randint(0, len(files) - 1)])

    def set_image_source(self, image_source: str):
        if not Generator.valid_dir(image_source):
            raise InvalidGeneratorArgumentException("Image source directory is not a directory or is empty.")
        if not Generator.dir_contains_only_imgs(image_source):
            raise InvalidGeneratorArgumentException("Image directory must only contain images (jpeg, png, ...).")
        self.image_source = image_source

    def set_img_output_dir(self, img_output_dir: str):
        if not os.path.isdir(img_output_dir):
            raise InvalidGeneratorArgumentException("No directory found under img_output_dir path.")
        self.img_output_dir = img_output_dir

    def set_num(self, num: int):
        if type(num) == int and num < 0:
            raise InvalidGeneratorArgumentException("Number of set output images must be positive.")
        self.num = num

    def set_augment_config(self, augment_config: configparser.ConfigParser):
        if augment_config == None:
            raise InvalidGeneratorArgumentException("No configparser object provided.")
        self.augment_config = augment_config

    @abstractmethod
    def generate(self):
        pass


class ClassificationGenerator(Generator):

    def __init__(self, image_source: str, img_output_dir: str, num: int, augment_config: configparser.ConfigParser):
        super().__init__(image_source, img_output_dir, num, augment_config)

    def generate(self):
        for i in range(self.num):
            image = cv2.imread(Generator.get_random_file(self.image_source))
            image, _ = augmentapplier.apply(image=image, augment_config=self.augment_config)
            file_name = os.path.basename(f'{os.path.normpath(self.img_output_dir)}_{str(i)}.jpg')
            cv2.imwrite(os.path.join(self.img_output_dir, file_name), image)
            print(file_name)
        print('Generating finished.')


'''
This class enables user to synthesize datasets for object detection training & testing.
It does so by randomly overlapping "object" images onto "background" images.
Additionally the resulting images will be augmented and annotated in different formats.
'''


class DetectionGenerator(Generator):

    def __init__(self, image_source: str, object_source: str, num: int, img_output_dir: str, annotation_format: str,
                 annotation_output_dir: str, max_objects: int, augment_config: configparser.ConfigParser):
        super().__init__(image_source, img_output_dir, num, augment_config)
        self.set_objects_source(object_source)
        self.set_annotation_format(annotation_format)
        self.set_annotation_output_dir(annotation_output_dir)
        self.set_max_objects(max_objects)

    @staticmethod
    def dir_contains_only_pngs(dir: str) -> bool:
        if Generator.dir_contains_only_imgs(dir):
            for file in os.listdir(dir):
                img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_UNCHANGED)
                # real PNG images have a shape of (x,x,4)
                if img.shape[2] != 4:
                    return False
            return True
        return False
    
    def set_objects_source(self, objects_source: str):
        if not Generator.valid_dir(objects_source):
            raise InvalidGeneratorArgumentException("Object source directory not a real directory or is empty.")
        for dir in os.listdir(objects_source):
            path = os.path.join(objects_source, dir)
            if not Generator.valid_dir(path):
                raise InvalidGeneratorArgumentException("Found invalid or empty sub-directory in object source.")
            if not DetectionGenerator.dir_contains_only_pngs(path):
                raise InvalidGeneratorArgumentException("Found invalid png image in object source sub-directory.")
        self.objects_source = objects_source

    def set_annotation_format(self, annotation_format):
        if not annotation_format.lower() in annotationwriters.ACCEPTED_ANNOTATION_FORMATS:
            raise InvalidGeneratorArgumentException("Specified annotation format not a valid format, accepted formats are xml & json")
        self.annotation_format = annotation_format

    def set_annotation_output_dir(self, output_dir: str):
        if not os.path.isdir(output_dir):
            raise InvalidGeneratorArgumentException("No directory found under specified annotation-output-path.")
        self.annotation_output_dir = output_dir

    def set_max_objects(self, max_objects: int):
        if max_objects < 1:
            raise InvalidGeneratorArgumentException("Numbe  r of max objects per image must be at least 1.")
        self.max_objects = max_objects

    def get_random_objects(self) -> tuple:
        objects = []
        labels = []
        dir_list = os.listdir(self.objects_source)
        for i in range(random.randint(1, self.max_objects)):
            idx = random.randint(0, len(dir_list) - 1)
            class_dir = os.path.join(self.objects_source, dir_list[idx])
            images = os.listdir(class_dir)
            objects.append(os.path.join(class_dir, images[random.randint(0, len(images) - 1)]))
            labels.append(dir_list[idx])
        return objects, labels

    def create_image(self, image, objects) :
        boxes = []
        image = cv2.imread(image)
        # copy paster objects onto background image
        for obj in objects:
            obj_image = cv2.imread(obj, cv2.IMREAD_UNCHANGED)
            image, box = BoxAugmentations.overlay_object(image, obj_image)
            boxes.append(box)   
        image, boxes = augmentapplier.apply(image=image, boxes=boxes, augment_config=self.augment_config)
        img_width = image[0].__len__()
        img_height = image.__len__()
        return image, boxes, img_width, img_height

    def generate(self):
        for i in range(self.num):
            image = Generator.get_random_file(self.image_source) 
            obj_images, labels = self.get_random_objects()
            image, boxes, img_width, img_height = self.create_image(image, obj_images)
            img_path = f'{self.img_output_dir}generated_{i}.jpg'
            cv2.imwrite(img_path, image)
            annotationwriters.write_annotations(self.annotation_format, img_path, img_width,
                                                img_height, labels, boxes, self.annotation_output_dir)
            print(img_path)
        print('Generating finished.')
