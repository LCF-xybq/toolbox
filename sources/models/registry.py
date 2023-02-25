from cv.cnn import MODELS as CV_MODELS
from cv.utils import Registry

MODELS = Registry('model', parent=CV_MODELS)
COMPONENTS = MODELS
LOSSES = MODELS
