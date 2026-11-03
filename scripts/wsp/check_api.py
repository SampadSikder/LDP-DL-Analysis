#!/usr/bin/env python
"""One small two-step tool call against the configured model (WSP_MODEL, WSP_REASONING_EFFORT from .env).

Checks: the key works from this machine, the model id and reasoning effort are accepted, a tool call and its
result round-trip (reasoning items passed back), and usage fields (cached / reasoning tokens) are present.
Tokens are recorded in the shared ledger like any other call. Prints no secrets.
"""
import sys
import time

from _common import load_env


def main():
    env_path = load_env()
    import os
    if not os.environ.get('OPENAI_API_KEY'):
        print(f'OPENAI_API_KEY not set (expected in {env_path})')
        return 1
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.tools import tool

    from attacker_detector.wsp.agent import make_llm
    from attacker_detector.wsp.cost import CostTracker, Ledger, Pricing

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    llm = make_llm()
    print(f"model {os.environ.get('WSP_MODEL')}, reasoning effort {os.environ.get('WSP_REASONING_EFFORT', 'xhigh')}")
    cost = CostTracker('check_api', Pricing.load(), Ledger())
    bound = llm.bind_tools([add])
    msgs = [HumanMessage('Use the add tool to compute 2 + 3, then answer with just the number.')]
    t0 = time.perf_counter()
    ai = bound.invoke(msgs)
    u, usd = cost.add_call(ai, time.perf_counter() - t0)
    print(f'call 1: tool_calls={[c["name"] for c in ai.tool_calls]} usage={u} ${usd:.4f}')
    if not ai.tool_calls:
        print('FAIL: the model did not call the tool')
        return 1
    msgs += [ai, ToolMessage(content=str(add.invoke(ai.tool_calls[0]['args'])), tool_call_id=ai.tool_calls[0]['id'])]
    t0 = time.perf_counter()
    ai2 = bound.invoke(msgs)
    u2, usd2 = cost.add_call(ai2, time.perf_counter() - t0)
    text = ai2.content if isinstance(ai2.content, str) else ' '.join(
        b.get('text', '') for b in ai2.content if isinstance(b, dict))
    print(f'call 2: answer={text.strip()[:80]!r} usage={u2} ${usd2:.4f}')
    ok = '5' in text
    print(('OK' if ok else 'CHECK ANSWER') + f": total {cost.llm['total']:,} tokens, ${cost.llm['usd']:.4f}; "
          f"reasoning tokens reported: {cost.llm['reasoning'] > 0}; cached field present: {'cache_read' in str(ai2.usage_metadata)}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
