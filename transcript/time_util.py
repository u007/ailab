import time

def measure_elapsed_time(func, *args, **kwargs):
    """
    Measure the elapsed time for a function call.

    Args:
        func (callable): The function to be called.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: A tuple containing the result of the function call and the elapsed time in seconds.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time