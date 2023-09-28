import argparse
import configparser
from pprint import pprint
import traceback
from generators import ClassificationGenerator, DetectionGenerator
import copy
from generatorexception import InvalidGeneratorArgumentException

parser = argparse.ArgumentParser(description='Augments images for image classifcation to improve dataset or/and it increase if')
parser.add_argument('-c', '--config', type=str, help='Path to config file controlling generator type and augmentations.', required=True)
args = parser.parse_args()
config_path = args.config
config = configparser.ConfigParser()

try:
    with open(config_path, 'r') as f:
        config.read_file(f) 
    augment_config = copy.deepcopy(config)
    augment_config.remove_section('generator')
    if config.get('generator', 'type') == 'classification':
        generator = ClassificationGenerator(image_source=config.get('generator', 'img_source'),
                                            img_output_dir=config.get('generator', 'img_output_dir'),
                                            num=int(config.get('generator', 'img_num')),
                                            augment_config=augment_config)
    elif config.get('generator', 'type') == 'detection':
        generator = DetectionGenerator(image_source=config.get('generator', 'img_source'),
                                        object_source=config.get('generator', 'obj_source'),
                                        img_output_dir=config.get('generator', 'img_output_dir'),
                                        annotation_format=config.get('generator', 'annotation_format'),
                                        annotation_output_dir=config.get('generator', 'annotation_output_dir'), 
                                        num=int(config.get('generator', 'img_num')),
                                        max_objects=int(config.get('generator', 'max_obj')),
                                        augment_config=config)
    else:
        raise InvalidGeneratorArgumentException('Specified generator type is not valid, please choose either \'classification\' or \'detection\'.')
    generator.generate()
except (FileNotFoundError, configparser.Error, InvalidGeneratorArgumentException) as exc:
    print(f'An error occurred: {type(exc).__name__} – {exc}')
except KeyError as exc:
    _, _, funcname, _ = traceback.extract_tb(exc.__traceback__)[-1]
    print(f'An error occurred: {type(exc).__name__} – missing argument {exc} in {funcname}')
except ValueError as exc:
    _, _, funcname, _ = traceback.extract_tb(exc.__traceback__)[-1]
    print(f'An error occurred: {type(exc).__name__} – in {funcname} {exc}')