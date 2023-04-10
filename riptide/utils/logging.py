import logging
import traceback

logging.basicConfig(
    level=logging.WARNING,
    format="{levelname:<8} {message}",
    style="{",
)


def logger():
    def decorator(f):
        def inner_func(*__args__, **__kwargs__):
            try:
                logging.info(f"Executing {f.__name__}")
                res = f(*__args__, **__kwargs__)
                logging.info(f"Successfully executed: {f.__name__}")
                return res
            except Exception as e:
                stack_trace = "\n".join(traceback.format_exc().splitlines()[3:])
                logging.error(f"Error in: {f.__name__}\n{stack_trace}", exc_info=False)

        return inner_func

    return decorator
