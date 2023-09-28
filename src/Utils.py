import os
import random
import types
import cv2


class DataUtils:

    # check if a directory exists and whether it is empty
    @staticmethod
    def valid_dir(dir: str) -> bool:
        if not os.path.isdir(dir) or len(os.listdir(dir)) == 0:
            return False
        return True

    @staticmethod
    def dir_contains_only_imgs(dir: str) -> bool:
        for file in os.listdir(dir):
            img = cv2.imread(os.path.join(dir, file))
            # if img is of type NoneType then it cannot be a valid image file (expected: numpy.ndarray)
            if type(img) == types.NoneType:
                print("Invalid image file found: " + str(file))
                return False
        return True

    @staticmethod
    def dir_contains_only_pngs(dir: str) -> bool:
        if DataUtils.dir_contains_only_imgs(dir):
            for file in os.listdir(dir):
                img = cv2.imread(os.path.join(dir, file), cv2.IMREAD_UNCHANGED)
                # real PNG images have a shape of (x,x,4)
                if img.shape[2] != 4:
                    print("Invalid png file found: " + str(file))
                    return False
            return True
        return False

    @staticmethod
    def get_random_file(dir: str) -> str:
        files = os.listdir(dir)
        return os.path.join(dir, files[random.randint(0, len(files) - 1)])



