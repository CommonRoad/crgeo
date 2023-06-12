import traceback
import sys
from typing import Tuple


def get_error_stack_summary() -> Tuple[str, int, str]:
    """
    Returns the latest/current exception info
    """
    _, _, var = sys.exc_info()
    tb_info = traceback.extract_tb(var)
    file_location, line_number, _, detail = tb_info[-1]

    return file_location, line_number, detail
