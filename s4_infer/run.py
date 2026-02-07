import sys
import s0_utils.global_params as g
from pathlib import Path
from s0_utils.Classifier import Classifier

path = "data_sets/test/Melodic Extratone/Aekhloria - Ashes of Nothing.mp3"
if len(sys.argv) > 1:
    path = sys.argv[1]

path = Path(path)

classifier = Classifier(g.NAME)
top, _ = classifier.infer(path)
if top is None or len(top) == 0:
    print(f'Inference failed on "{path}"!')
    sys.exit(1)

print(path.name)
classifier.print_top(top)
