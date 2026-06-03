import os
import sys

# Make the project modules (monitor, config) importable from the tests.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
