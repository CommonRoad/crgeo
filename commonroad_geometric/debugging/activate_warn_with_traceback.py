"""Enables traceback logging for warnings"""
import sys
import traceback
import warnings


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    import warnings
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def activate_warn_with_traceback():
    warnings.showwarning = warn_with_traceback
