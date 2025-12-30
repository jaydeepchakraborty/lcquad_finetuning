
class LCQUADException(Exception):
    def __init__(self, ex, message):
        if ex is not None:
            self.message = ex.message
        else:
            self.message = message

    def __str__(self):
        return f"{self.message})"