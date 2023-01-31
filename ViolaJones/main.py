from utils import load_pictures, load_annotations
from matplotlib import pyplot as plt

pictures = load_pictures()
annotations = load_annotations()
key = list(pictures.keys())[0]
picture = pictures[key]
annotation = annotations[key]["category"]
print(annotation)