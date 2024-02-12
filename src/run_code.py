import multiprocessing
import threading


def timeout_handler(process):
    if process.is_alive():
        print("Timeout reached! Cancelling the function.")
        process.terminate()


def run_function_with_timeout(target_function, args=(), timeout_seconds=5):
    # Initialize return_dict for us to store values in
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict["pass_tests"] = False

    # Create a wrapper function to handle storing the right value in return_dict
    def target_function_wrapper(target_function, return_dict, *args):
        try:
            target_function(*args)
            return_dict["pass_tests"] = True
        except Exception as e:
            return_dict["pass_tests"] = False

    # Define the process
    process = multiprocessing.Process(
        target=target_function_wrapper, args=(target_function, return_dict) + args
    )

    # Start the process
    process.start()

    # Set up a timer to terminate the process if it exceeds the timeout
    timer = threading.Timer(timeout_seconds, timeout_handler, args=(process,))
    timer.start()

    # Wait for the process to finish
    process.join()

    # Cancel the timer
    timer.cancel()
    return return_dict.values()[0]


def does_code_pass_tests(generated_function, tests):
    code = generated_function + tests
    pass_tests = run_function_with_timeout(exec, args=(code,))
    return pass_tests
