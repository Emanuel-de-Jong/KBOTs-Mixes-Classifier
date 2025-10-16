from datetime import datetime

class Logger():
    def __init__(self, file_path, should_append=True):
        self.file_path = file_path
        self.should_append = should_append
    
    def writeln(self, msg):
        with open(self.file_path, "a" if self.should_append else "w") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} {msg}\n")
        print(msg)
