class AdapterException(Exception):
    pass


class AdapterModelNotFoundException(AdapterException):
    pass


class AdapterRateLimitException(AdapterException):
    pass


_all_ = [
    "AdapterException",
    "AdapterModelNotFoundException",
    "AdapterRateLimitException",
]
