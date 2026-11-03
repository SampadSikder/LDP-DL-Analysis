"""LangGraph agent: one tool-calling LLM loop with budget checks and deterministic history compaction.

Graph:  START -> llm -> (tools -> llm)* -> END
  llm    checks every stop condition, compacts history, calls the model, records usage
  tools  runs the tool calls in order, saves run state
Compaction keeps the system prompt, the first message and the last `keep_turns` turns; older turns are
replaced by a harness-written summary (edit log, probes, committed metrics, budget). No LLM call is spent
on summarizing, and every request stays under `max_request_tokens`.
"""

import inspect
import json
import os
import time
import traceback
from typing import Optional, TypedDict

from .log import dumps

SYSTEM_PROMPT = """You are a weight-space programmer. You improve an already trained neural network by reading \
and editing its weights directly. There is no optimizer, no gradient and no training data in your sandbox: you \
understand the network from its weights and from a limited number of forward-pass probes, then write targeted \
edits as numpy code.

TASK
{task}

ARCHITECTURE
{architecture}

INPUTS / OUTPUTS
inputs: {inputs}
outputs: {outputs}

METRIC
{metric}
Gate guards: {guards}

HOW AN EDIT IS JUDGED (plain code, not you)
- Staged weights W are compared with the committed weights. Caps per edit: at most {max_changed_frac:.0%} of all \
weights changed, relative L2 change at most {max_rel_l2}. Shapes and names must not change.
- The gate scores the edit on VALIDATION data. It is kept only if the {confidence:.0%} one-sided bootstrap bound \
says the loss improves, and no guard metric is worse by more than {guard_tol:.0%}.
- Probes run on a fixed subset of TRAINING data. Probe results are cheaper evidence; the gate decides.

BUDGET
{probes} probes, {gate_evals} gate evaluations, about {token_cap} tokens. The run stops when the gate budget is \
spent, after {stop_after_rejects} rejected edits in a row, or when you call finish.

WORKING STYLE
1. Read the model card, then inspect weights (tensor_stats, read_tensor, python) and probe the committed model to \
find concrete failure modes (which slices, which direction of error).
2. Form a hypothesis about which units / directions cause a failure. Check it with numpy on the weights.
3. Stage a small, targeted edit in W (python), probe it with weights='staged', and compare with committed.
4. Submit only edits the probes support. After a rejection, read the reasons and adjust or try another idea.
5. Keep tool outputs small (ask for slices, not whole tensors). Finish when further gains are unlikely.
Never claim results you have not probed or gated."""

FIRST_MESSAGE = """Model card and committed validation metrics:
{card}

Start by locating where the committed model fails, then edit."""

NUDGE = ('Use a tool: continue with probes / edits, submit_edit a staged edit, or call finish(summary) if you '
         'are done.')


def system_prompt(state):
    c, g, b = state.card, state.gate_rules, state.budget
    return SYSTEM_PROMPT.format(task=c['task'], architecture=c['architecture'], inputs=json.dumps(c['inputs']),
                                outputs=c['outputs'], metric=c['metric'], guards=c.get('gate_guards', ''),
                                max_changed_frac=g['max_changed_frac'], max_rel_l2=g['max_rel_l2'],
                                confidence=g['confidence'], guard_tol=g['guard_tol'], probes=b.probes,
                                gate_evals=b.gate_evals, token_cap=f'{b.token_cap:,}' if b.token_cap else 'unlimited',
                                stop_after_rejects=b.stop_after_rejects)


def first_message(state):
    card = {k: v for k, v in state.card.items() if k not in ('task', 'architecture', 'metric')}
    card['committed_validation'] = state.cur_val.report
    card['stage1_validation'] = state.val_stage1
    card['budget'] = state.budget.status()
    msg = FIRST_MESSAGE.format(card=dumps(card))
    if state.edits:
        msg += '\n\nThis run is resuming after an interruption. Earlier work:\n' + dumps(state.history())
    return msg


SUMMARY_TAG = '[harness summary of earlier turns, written by code]'


def summary_message(state):
    return (SUMMARY_TAG + '\n' + dumps(state.history())
            + '\nCommitted weights include every accepted edit above; your sandbox W was kept as it was.')


def make_llm(env=None):
    from langchain_openai import ChatOpenAI
    env = env or os.environ
    model = env.get('WSP_MODEL')
    if not model:
        raise SystemExit('WSP_MODEL is not set (put it in .env)')
    reasoning = {'effort': env.get('WSP_REASONING_EFFORT', 'xhigh')}
    if env.get('WSP_REASONING_SUMMARY'):
        reasoning['summary'] = env['WSP_REASONING_SUMMARY']
    kw = dict(model=model, use_responses_api=True, reasoning=reasoning,
              max_retries=int(env.get('WSP_MAX_RETRIES', 6)), timeout=float(env.get('WSP_TIMEOUT_S', 1200)))
    if env.get('WSP_BASE_URL'):
        kw['base_url'] = env['WSP_BASE_URL']
    if env.get('WSP_OUTPUT_VERSION'):
        kw['output_version'] = env['WSP_OUTPUT_VERSION']
    if env.get('WSP_SERVICE_TIER'):
        kw['service_tier'] = env['WSP_SERVICE_TIER']
    if env.get('WSP_MAX_OUTPUT_TOKENS'):
        kw['max_tokens'] = int(env['WSP_MAX_OUTPUT_TOKENS'])
    return ChatOpenAI(**kw)


def _turns(msgs):
    """Group messages after the first two into turns: an AI message with its tool results, or a human message."""
    from langchain_core.messages import AIMessage, ToolMessage
    turns = []
    for m in msgs:
        if isinstance(m, ToolMessage) and turns:
            turns[-1].append(m)
        elif isinstance(m, AIMessage) or not turns or not isinstance(m, ToolMessage):
            turns.append([m])
    return turns


class Agent:
    def __init__(self, state, llm, tools, fns, cfg):
        self.state, self.llm, self.fns = state, llm, fns
        self.bound = llm.bind_tools(tools)
        self.cfg = cfg['agent']
        self.last_input = 0
        self.error = None

    def _compact(self, msgs, force=False):
        from langchain_core.messages import HumanMessage
        head, body = msgs[:2], msgs[2:]
        turns = [t for t in _turns(body) if not str(getattr(t[0], 'content', '')).startswith(SUMMARY_TAG)]
        k = self.cfg['keep_turns']
        big = self.last_input > self.cfg['compact_at_tokens']
        if self.last_input > 0.8 * self.cfg['max_request_tokens']:
            k = max(2, k // 3)
        if not (force or big or len(turns) > k + 4):
            return msgs
        keep = [m for t in turns[-k:] for m in t]
        summ = HumanMessage(summary_message(self.state))
        self.state.log.event('compaction', dropped_turns=len(turns) - min(k, len(turns)),
                             last_input_tokens=self.last_input)
        return head + [summ] + keep

    def _llm_node(self, s):
        from langchain_core.messages import HumanMessage
        st = self.state
        reason = 'finished' if st.finished else st.budget.llm_stop_reason()
        if reason:
            return {'stop': reason}
        msgs = self._compact(s['messages'])
        t0 = time.perf_counter()
        try:
            ai = self.bound.invoke(msgs)
        except Exception as e:                  # noqa: BLE001 - API errors end the run; it can resume
            self.error = f'{type(e).__name__}: {e}'
            st.log.event('llm_error', error=self.error[:2000])
            return {'messages': msgs, 'stop': 'llm_error'}
        wait = time.perf_counter() - t0
        u, usd = st.cost.add_call(ai, wait)
        st.budget.charge_call(u['total'])
        self.last_input = u['input']
        st.log.message('assistant', ai, usage=u, usd=usd, wait_s=round(wait, 2))
        msgs = msgs + [ai]
        if not ai.tool_calls:
            n = s.get('nudges', 0) + 1
            if n > 2:
                return {'messages': msgs, 'stop': 'no_tool_calls', 'nudges': n}
            return {'messages': msgs + [HumanMessage(NUDGE)], 'nudges': n}
        return {'messages': msgs, 'nudges': 0}

    def _tools_node(self, s):
        from langchain_core.messages import ToolMessage
        st = self.state
        ai = s['messages'][-1]
        out = []
        for tc in ai.tool_calls:
            fn = self.fns.get(tc['name'])
            args = tc.get('args') or {}
            t0 = time.perf_counter()
            if fn is None:
                res = f"error: unknown tool {tc['name']!r}"
            else:
                try:
                    inspect.signature(fn).bind(**args)
                except TypeError as e:
                    res = f'error: bad arguments: {e}'
                else:
                    try:
                        res = fn(**args)
                    except Exception as e:      # noqa: BLE001 - tool failures go back to the agent
                        res = f'error: {type(e).__name__}: {e}'
                        st.log.event('tool_exception', name=tc['name'], error=traceback.format_exc()[-2000:])
            st.log.event('tool', name=tc['name'], args=tc.get('args'), result=res,
                         seconds=round(time.perf_counter() - t0, 2))
            out.append(ToolMessage(content=res, tool_call_id=tc['id'], name=tc['name']))
        st.save()
        stop = 'finished' if st.finished else st.budget.stop_reason()
        return {'messages': s['messages'] + out, 'stop': stop}

    def run(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.graph import END, START, StateGraph

        class S(TypedDict, total=False):
            messages: list
            stop: Optional[str]
            nudges: int

        g = StateGraph(S)
        g.add_node('llm', self._llm_node)
        g.add_node('tools', self._tools_node)
        g.add_edge(START, 'llm')

        def after_llm(s):
            if s.get('stop'):
                return END
            last = s['messages'][-1]
            return 'tools' if getattr(last, 'tool_calls', None) else 'llm'

        g.add_conditional_edges('llm', after_llm, ['tools', 'llm', END])
        g.add_conditional_edges('tools', lambda s: END if s.get('stop') else 'llm', ['llm', END])
        graph = g.compile()
        msgs = [SystemMessage(system_prompt(self.state)), HumanMessage(first_message(self.state))]
        self.state.log.event('prompt', system=msgs[0].content, first=msgs[1].content)
        limit = 2 * int(self.cfg.get('max_llm_calls', 200)) + 50
        out = graph.invoke({'messages': msgs, 'stop': None, 'nudges': 0}, {'recursion_limit': limit})
        return out.get('stop') or 'ended'
