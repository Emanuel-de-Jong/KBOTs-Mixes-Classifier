import sys
import global_params as g
from Classifier import Classifier

path = "test/Bassline/Club Angel - Control Dem.mp3"
if len(sys.argv) > 1:
    path = sys.argv[1]

classifier = Classifier("global")
top, _ = classifier.infer(path)
if top is None or len(top) == 0:
    print(f'Inference failed on "{path}"!')
    sys.exit(1)

classifier.print_top(top)
