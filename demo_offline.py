"""Offline CP7 demo: scripted model behavior with real bundled snapshots; no API calls."""
import json

from agent import run_agent
from status_tools import StatusTools


def _tool_call(call_id, name, arguments):
    args = json.dumps(arguments, ensure_ascii=False)
    return {
        'message': {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {'name': name, 'arguments': args},
            }],
        },
        'calls': [{'id': call_id, 'name': name, 'arguments': args}],
        'finish': 'tool_calls',
        'usage': {},
    }


def _final(text):
    return {
        'message': {'role': 'assistant', 'content': text},
        'calls': [],
        'finish': 'stop',
        'usage': {},
    }


class SnapshotDemoProvider:
    """Deterministically asks for status, then incident details, then answers."""

    def __init__(self):
        self.step = 0

    def complete(self, messages, schemas):
        self.step += 1
        if self.step == 1:
            return _tool_call('demo-status', 'get_status', {
                'provider': 'cloudflare', 'component': 'Cache Purge',
            })
        if self.step == 2:
            status = json.loads(messages[-1]['content'])
            incidents = status.get('incidents', [])
            if not incidents:
                raise RuntimeError('Snapshot has no matching Cache Purge incident.')
            incident = incidents[0]
            return _tool_call('demo-details', 'get_incident_details', {
                'provider': 'cloudflare', 'incident_id': incident['id'],
            })
        if self.step == 3:
            details = json.loads(messages[-1]['content'])
            if not details.get('ok'):
                return _final('I could not verify an incident from the snapshot.')
            updates = details.get('updates', [])
            latest = updates[0].get('body', '')[:260] if updates else 'No update text was returned.'
            answer = (
                f"Snapshot captured at {details['captured_at']}: "
                f"{details['name']} is {details['status']} (updated {details['updated_at']}). "
                f"Latest available update: {latest} Source: {details['url']}"
            )
            return _final(answer)
        raise RuntimeError('Unexpected extra model step in offline demo.')


class TruncatedDemoProvider:
    """Shows that a truncated model response is returned as an error."""

    def complete(self, messages, schemas):
        return {
            'message': {'role': 'assistant', 'content': 'Partial answer, not complete'},
            'calls': [],
            'finish': 'length',
            'usage': {},
        }


def _show(title, result):
    print(f'\n=== {title} ===')
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    question = 'Check Cloudflare Cache Purge and inspect one matching incident with evidence.'
    multi_step = run_agent(question, SnapshotDemoProvider(), StatusTools('snapshot'), max_steps=5)
    _show('MULTI-STEP SNAPSHOT TRACE (scripted provider; not an LLM evaluation)', multi_step)

    failure = run_agent(
        'Demonstrate handling of a truncated model response.',
        TruncatedDemoProvider(),
        StatusTools('snapshot'),
        max_steps=2,
    )
    _show('ERROR TRACE (simulated finish=length)', failure)

    return 0 if multi_step['status'] == 'completed' and failure['status'] == 'error' else 1


if __name__ == '__main__':
    raise SystemExit(main())
