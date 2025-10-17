import torch
from datetime import datetime
from Mert import Mert

class Logger():
    def __init__(self, file_path):
        self.file_path = file_path
    
    def writeln(self, msg):
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} {msg}\n")
        print(msg)

def preprocess(embs):
    for i in range(len(embs)):
        emb = embs[i]
        embs[i] = emb

    # X_train_norm = torch.nn.functional.normalize(torch.from_numpy(X_train), p=2, dim=1)
    # X_test_norm = torch.nn.functional.normalize(torch.from_numpy(X_test), p=2, dim=1)
    # X_train_norm = X_train_norm.numpy()
    # X_test_norm = X_test_norm.numpy()

    embs = (embs - embs.mean(axis=0)) / embs.std(axis=0)
    embs = embs.transpose(0, 2, 3, 1)

    return embs
