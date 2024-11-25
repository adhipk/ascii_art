import time
from functools import wraps
from collections import defaultdict

class TimerStats:
    def __init__(self):
        self.times = defaultdict(list)
    
    def add_time(self, method_name, elapsed):
        self.times[method_name].append(elapsed)
    
    def print_stats(self):
        print("\nPerformance Statistics:")
        print("-" * 50)
        for method, times in self.times.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            print(f"{method:30} Avg: {avg_time*1000:8.2f}ms Total: {total_time*1000:8.2f}ms Calls: {len(times)}")

def timer(timer_stats):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            timer_stats.add_time(func.__name__, elapsed)
            return result
        return wrapper
    return decorator
