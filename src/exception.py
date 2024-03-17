import sys

def message_detail(error, error_detail):
    _, _, tb = error_detail.exc_info()
    filename = tb.tb_frame.f_code.co_filename
    error_message = f"Error: {error} in {filename}"

    return error_message

class Error(Exception):
    def __init__(self, error, error_detail):
        self.error = error
        self.error_detail = error_detail

    def __str__(self):
        return message_detail(self.error, self.error_detail)
