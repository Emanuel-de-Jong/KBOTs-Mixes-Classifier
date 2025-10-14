import sys
from Classifier import Classifier

path = "test/Bassline/Club Angel - Control Dem.mp3"
if len(sys.argv) > 1:
    path = sys.argv[1]

classifier = Classifier()
top = classifier.infer(path)
if top is None or len(top) == 0:
    print("Critical problem!")
    sys.exit(0)

classifier.print_top(top)
