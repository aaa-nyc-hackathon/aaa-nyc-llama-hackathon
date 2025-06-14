from time import time

def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        result = {
            "time_delta": end - start,
            "data": result
        }