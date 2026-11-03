"""Shared setup for the WSP scripts: repo on sys.path, .env loaded (never printed)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def load_env():
    path = os.path.join(ROOT, '.env')
    if os.path.exists(path):
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
    return path
