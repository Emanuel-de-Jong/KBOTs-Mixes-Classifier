from datetime import datetime

class Logger():
    def __init__(self, file_path):
        self.file_path = file_path
    
    def writeln(self, msg):
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} {msg}\n")
        print(msg)
