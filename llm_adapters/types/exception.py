class AdapterException(Exception):
    pass


class ModelNotFoundException(AdapterException):
    pass


class RateLimitException(AdapterException):
    pass


_all_ = [
    "AdapterException",
    "ModelNotFoundException",
    "RateLimitException",
]
