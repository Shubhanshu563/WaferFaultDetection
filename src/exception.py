import sys

def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message including the script name, line number, and error message.
    :param error: The exception object.
    :param error_detail: The sys module for accessing exception details.
    :return: A formatted string containing the error details.
    """
    _, _, exc_tb = error_detail.exc_info()  # Get traceback object from sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # Extract the script name from the traceback
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Format error details into a readable string
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class that provides detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException instance.
        :param error_message: A brief error message string.
        :param error_detail: The sys module for accessing exception details.
        """
        super().__init__(error_message)  # Initialize the base Exception class
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail  # Generate detailed error message
        )

    def __str__(self):
        """
        Returns the detailed error message when the exception is printed or converted to a string.
        """
        return self.error_message
