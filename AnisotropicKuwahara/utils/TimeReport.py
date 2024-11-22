import time

"""used to time the runtime of something

example usage: 

with TimeReport("test_name"):
    # write code here

# once the code block is exited, the runtime will be printed to the console
"""

class TimeReport:
    def __init__(self, test_name):
        self.test_name = test_name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time
        print(f"{self.test_name} runtime: {self.runtime:.4f} seconds")