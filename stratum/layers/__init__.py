from stratum.layers.layer1 import Layer1
from stratum.layers.layer2 import Layer2
from stratum.layers.layer3 import Layer3
from stratum.layers.layer4 import Layer4
from stratum.layers.layer5 import Layer5

LAYER_REGISTRY = {1: Layer1, 2: Layer2, 3: Layer3, 4: Layer4, 5: Layer5}

__all__ = ["LAYER_REGISTRY", "Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]
