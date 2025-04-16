from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
    "common/materials.xml",
    "common/skybox.xml",
    "common/visual.xml",
]

def read_resource(file_path):
    full_path = os.path.abspath(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"文件未找到: {full_path}")
    with open(full_path, 'r') as file:
        return file.read()

ASSETS = {filename: read_resource(os.path.join(_SUITE_DIR, filename))
          for filename in _FILENAMES}

def read_model(model_filename):
    return read_resource(os.path.join(_SUITE_DIR, model_filename))
