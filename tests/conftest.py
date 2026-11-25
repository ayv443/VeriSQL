import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "verisql"))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIXTURES = os.path.join(ROOT, "eval", "fixtures")
