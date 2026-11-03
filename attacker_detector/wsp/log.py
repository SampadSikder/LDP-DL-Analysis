"""Run transcript: every prompt, reply, tool call, code cell, output and gate verdict, as JSON lines."""

import json
import os
import re
import time

import numpy as np

_KEY = re.compile(r'sk-[A-Za-z0-9_\-]{16,}')


def redact(text):
    if not isinstance(text, str):
        return text
    key = os.environ.get('OPENAI_API_KEY')
    if key and len(key) > 8:
        text = text.replace(key, '***')
    return _KEY.sub('sk-***', text)


def jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f'<array {o.shape}>'
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def dumps(obj, **kw):
    return json.dumps(obj, default=jsonable, **kw)


class RunLog:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def event(self, etype, **data):
        rec = {'t': round(time.time(), 3), 'type': etype, **data}
        with open(self.path, 'a') as f:
            f.write(redact(dumps(rec)) + '\n')

    def message(self, role, msg, **extra):
        content = msg.content if hasattr(msg, 'content') else msg
        rec = {'role': role, 'content': content}
        tc = getattr(msg, 'tool_calls', None)
        if tc:
            rec['tool_calls'] = [{'name': c['name'], 'args': c['args'], 'id': c.get('id')} for c in tc]
        ak = getattr(msg, 'additional_kwargs', None) or {}
        if ak.get('reasoning'):
            rec['reasoning'] = ak['reasoning']
        self.event('message', **rec, **extra)
