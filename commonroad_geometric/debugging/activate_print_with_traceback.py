"""Enables traceback logging for print statements"""
import traceback
import sys


class TracePrints(object):
    def __init__(self):    
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)

    
def activate_print_with_traceback():
    sys.stdout = TracePrints()
