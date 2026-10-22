"""Offline local-release tests. Run from the export root with pytest.

Load the actual server without its startup/session hydration block. All game
data lives in tmp_path; process creation and network access are forbidden.
"""
import ast
import builtins
import json
import os
import re
import socket
import subprocess
import symtable
import sys
import types
from functools import lru_cache
from pathlib import Path
from unittest.mock import Mock

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "social_turing_test"


RETIRED_KEYS = (
    'live_evaluator_model', 'live_evaluator_feedback_enabled',
    'live_evaluator_min_new_messages', 'live_evaluator_phase2_threshold',
    'live_evaluator_phase2_min_gap_s', 'live_evaluator_future_setting',
)


def assert_current_config(config):
    """Check current configuration only, never historical data or prose."""
    if isinstance(config, dict):
        assert not any(isinstance(key, str) and key.startswith('live_evaluator_')
                       for key in config)
        for value in config.values():
            assert_current_config(value)
    elif isinstance(config, (list, tuple)):
        for value in config:
            assert_current_config(value)


@pytest.fixture(autouse=True)
def offline_only(monkeypatch, tmp_path):
    """Never inherit provider credentials or allow real I/O in this module."""
    monkeypatch.syspath_prepend(str(REPOSITORY))
    for key in tuple(os.environ):
        if (key.endswith(('_API_KEY', '_API_TOKEN', '_ACCESS_TOKEN'))
                or key.startswith(('GOOGLE_', 'GCLOUD_', 'CLOUDSDK_', 'VERTEX_',
                                   'AWS_', 'AZURE_'))):
            monkeypatch.delenv(key)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', str(tmp_path / 'no-adc.json'))
    monkeypatch.setenv('CLOUDSDK_CONFIG', str(tmp_path / 'no-cloud-config'))

    def forbidden(*args, **kwargs):
        raise AssertionError('Credentials/network/process access is forbidden in offline tests')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    monkeypatch.setattr(socket.socket, 'sendto', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(os, 'system', forbidden)


@lru_cache(maxsize=1)
def server_code():
    """Cache code, not mutable Flask/session state, across parameter cases."""
    source = ROOT / 'web_player.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    # The final main/else block starts a server or restores real sessions. Neither
    # belongs in a unit test; all route/function definitions are executed intact.
    assert isinstance(tree.body[-1], ast.If)
    assert '__name__' in ast.unparse(tree.body[-1].test)
    tree.body.pop()
    return compile(tree, str(source), 'exec')


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(REPOSITORY))
    monkeypatch.setenv('STT_ADMIN_PASSWORD', 'offline-test-password')
    source = ROOT / 'web_player.py'
    module = types.ModuleType('local_release_web_test')
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(server_code(), module.__dict__)
    module.app.config.update(TESTING=True, SECRET_KEY='offline-only')
    module.app.template_folder = str(ROOT / 'templates')
    monkeypatch.setattr(module, '_persist_sessions', lambda: None)
    monkeypatch.setattr(module, 'resolve_game_dir', lambda gid: tmp_path / gid)
    monkeypatch.setattr(module, '_launch_lobby_ai', lambda *args: None)
    module._offline_root = tmp_path
    return module


@pytest.fixture
def client(server):
    client = server.app.test_client()
    response = client.post('/api/admin/login', json={'password': 'offline-test-password'})
    assert response.status_code == 200
    return client


def make_game(server, count=0, ai=True, opened=True):
    game = server._offline_root / '0001'
    game.mkdir(exist_ok=True)
    server.ensure_game_subdirs(game)
    settings = {
        'async_type': 'social_turing_test_agent', 'orchestration_mode': 'synchronous',
        'agent_code': 'current',
    }
    players = ([{'name': 'Player1', 'real_name': 'LLM0', 'is_ai': True,
                 'is_llm': True, 'llm_config': settings}] if ai else [])
    humans = [{'codename': f'Human{i}', 'real_name': f'Person {i}',
               'session_id': f'test-{i}', 'instructions_ack': False}
              for i in range(count)]
    if not opened:
        players += [{'name': h['codename'], 'real_name': h['real_name'],
                     'is_llm': False, 'is_ai': False, 'llm_config': {}}
                    for h in humans]
    (game / server.GAME_CONFIG_FILE).write_text(
        json.dumps({'players': players, 'daytime_minutes': 2.5}), encoding='utf-8')
    (game / server.GAME_START_TIME_FILE).touch()
    server._save_lobby(game, {'open': opened, 'ai_codename': 'Player1' if ai else None,
                              'humans': humans})
    return game


def test_password_is_env_only(server, monkeypatch):
    monkeypatch.delenv('STT_ADMIN_PASSWORD')
    assert server._load_admin_password() is None
    monkeypatch.setattr(server, 'ADMIN_PASSWORD', None)
    assert server.app.test_client().post('/api/admin/login', json={}).status_code != 200
    monkeypatch.setenv('STT_ADMIN_PASSWORD', 'own-password')
    assert server._load_admin_password() == 'own-password'
    assert '_preload_api_keys_from_secrets' not in server.__dict__


@pytest.mark.parametrize(("arguments", "host"), [([], "127.0.0.1"), (["--lan"], "0.0.0.0")])
def test_local_launcher_bind_and_session_restore(server, monkeypatch, arguments, host):
    run = Mock()
    restore = Mock()
    monkeypatch.setattr(server.app, "run", run)
    monkeypatch.setattr(server, "_load_persisted_sessions", restore)
    monkeypatch.setattr(server, "REPOSITORY_ROOT", server._offline_root)
    server.main([*arguments, "--port", "5055"])
    run.assert_called_once_with(debug=False, host=host, port=5055)
    restore.assert_called_once_with()
    assert Path.cwd() == server._offline_root


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_local_launcher_rejects_invalid_ports(server, monkeypatch, port):
    run = Mock()
    monkeypatch.setattr(server.app, "run", run)
    with pytest.raises(SystemExit):
        server.main(["--port", port])
    run.assert_not_called()


def test_packaged_paths_and_deployment_absence(server):
    from social_turing_test.paths import CONFIGURATIONS, PACKAGE_ROOT, REPOSITORY_ROOT

    assert PACKAGE_ROOT == ROOT
    assert REPOSITORY_ROOT == REPOSITORY
    assert (CONFIGURATIONS / "conversation_topics.json").is_file()
    assert (CONFIGURATIONS / "model_knowledge_cutoffs.json").is_file()
    assert server._BASE_DIR == str(REPOSITORY)
    assert Path(server.DIRS_PREFIX).parts == ("local_data", "games")
    assert server._SESSIONS_FILE == Path("local_data/sessions.json")
    for name in ("Dockerfile", ".dockerignore", "requirements-server.txt"):
        assert not (REPOSITORY / name).exists()
    assert "gunicorn" not in (REPOSITORY / "requirements.txt").read_text(encoding="utf-8").lower()
    tree = ast.parse((ROOT / "web_player.py").read_text(encoding="utf-8"))
    launches = [
        node.args[0] for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
    ]
    assert len(launches) == 3
    for command in launches:
        assert isinstance(command, ast.List)
        assert command.elts[1].value == "-m"
        assert command.elts[2].value in {
            "social_turing_test.llm_interface", "social_turing_test.stt_main"
        }


def test_fingerprint_covers_local_package_only():
    from social_turing_test.core.source_fingerprint import _candidate_files

    paths = {p.relative_to(REPOSITORY).as_posix() for p in _candidate_files(REPOSITORY)}
    assert {"run_game.py", "requirements.txt", "social_turing_test/web_player.py",
            "social_turing_test/export_game.py",
            "social_turing_test/configurations/model_knowledge_cutoffs.json",
            "social_turing_test/templates/game.html"} <= paths
    assert all(p.startswith("social_turing_test/") or p in {"run_game.py", "requirements.txt"}
               for p in paths)


def test_local_raw_file_routes_preserve_runtime_boundary_and_auth(server, client, monkeypatch):
    monkeypatch.setattr(server, "_BASE_DIR", str(server._offline_root))
    game = server._offline_root / "local_data" / "games" / "real" / "0001"
    game.mkdir(parents=True)
    monkeypatch.setattr(server, "resolve_game_dir", lambda _: game)
    (game / "phase_status.txt").write_text("daytime", encoding="utf-8")
    (game / "config.json").write_text("{}", encoding="utf-8")
    public = server.app.test_client()
    assert public.get("/api/file", query_string={"path": "games/0001/phase_status.txt"}).status_code == 200
    assert public.get("/api/file", query_string={"path": "games/0001/config.json"}).status_code == 401
    assert client.get("/api/file", query_string={"path": "games/0001/config.json"}).status_code == 200
    listed = public.get("/api/list_files", query_string={"path": "games/0001"})
    assert listed.status_code == 200
    assert listed.get_json()["files"] == ["phase_status.txt"]
    for endpoint in ("/api/file", "/api/list_files"):
        assert client.get(endpoint, query_string={"path": "data/games/game_001"}).status_code == 403
        assert client.get(endpoint, query_string={"path": "games/0001/../../../outside"}).status_code == 403


def test_index_and_excluded_routes(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Local / manual' in response.data
    for route in ('/prolific-entry', '/prolific_join', '/dashboard',
                  '/api/admin/evaluate_game', '/api/admin/eval_status'):
        response = client.post(route, json={}) if route.endswith(('join', 'evaluate_game')) else client.get(route)
        assert response.status_code == 404, route


def test_defaults_and_no_manifest_requirement(server, client):
    response = client.post('/api/admin/create_config', json={})
    assert response.status_code == 200
    config = response.get_json()['config']
    assert config['daytime_minutes'] == 2.5
    assert not config.get('max_humans')
    settings = config['players'][0]['llm_config']
    assert settings['model_name'] == 'gemini-2.5-flash'
    assert settings['scheduler_model'] == 'gemini-2.5-flash'
    assert settings['reviewer_model'] == 'gemini-2.5-flash'
    assert settings['persona'] == 'swiss_student'
    assert settings['num_candidates'] == 4
    assert settings['pipeline_timeout_s'] == 35
    assert settings['voting_strategy'] == 'strategic'
    assert_current_config(config)
    assert 'gemini-2.5-pro' not in json.dumps(config)
    assert server._local_config_error(config) is None


@pytest.mark.parametrize('provider', ['together', 'openrouter'])
def test_model_controls_and_timing_remain_editable(client, provider):
    response = client.post('/api/admin/create_config', json={
        'generator_provider': provider, 'generator_model': 'custom/generator',
        'review_mode': provider, 'reviewer_model': 'custom/reviewer',
        'scheduler_model': 'custom/scheduler', 'sub_agent_model': 'custom/worker',
        'daytime_minutes': 4,
        'persona': 'swiss_student',
        'advanced_llm_config': {'num_candidates': 6, 'pipeline_timeout_s': 40},
    })
    assert response.status_code == 200
    config = response.get_json()['config']
    settings = config['players'][0]['llm_config']
    assert config['daytime_minutes'] == 4
    assert settings['reviewer_backend'] == provider
    assert settings['use_' + provider] is True
    assert_current_config(config)
    for key, value in [('model_name', 'custom/generator'), ('reviewer_model', 'custom/reviewer'),
                       ('scheduler_model', 'custom/scheduler'), ('sub_agent_model', 'custom/worker'),
                       ('num_candidates', 6),
                       ('pipeline_timeout_s', 40)]:
        assert settings[key] == value


def test_unsupported_and_advanced_pipeline_bypasses(client):
    for pipeline in ('unsupported', 'other', 'experimental', 'unknown'):
        assert client.post('/api/admin/create_config', json={'pipeline': pipeline}).status_code == 400
    assert client.post('/api/admin/create_config', json={'agent_code': 'unsupported'}).status_code == 400
    response = client.post('/api/admin/create_config', json={'advanced_llm_config': {
        'async_type': 'schedule_then_generate', 'orchestration_mode': 'active',
        'allow_active_authoritative': True, 'agent_code': 'unsupported',
        'multi_candidate_generation': False, 'investigator': False,
        'voting_strategy': 'random',
    }})
    assert response.status_code == 200
    settings = response.get_json()['config']['players'][0]['llm_config']
    assert settings['async_type'] == 'social_turing_test_agent'
    assert settings['orchestration_mode'] == 'synchronous'
    assert settings['agent_code'] == 'current'
    assert not settings['allow_active_authoritative']
    assert settings['multi_candidate_generation'] is True
    assert settings['investigator'] is True
    assert settings['voting_strategy'] == 'strategic'
    assert_current_config(settings)


def test_unlimited_joins_unique_fallback_and_duplicate_names(server, client, monkeypatch):
    game = make_game(server)
    monkeypatch.setattr(server, 'OPTIONAL_CODE_NAMES', ['Player1', 'Neutral'])
    names = []
    for i in range(8):
        response = client.post('/lobby_join', json={'game_id': '0001', 'real_name': f'Person {i}'})
        assert response.status_code == 200
        names.append(response.get_json()['code_name'])
    assert len(set(names + ['Player1'])) == 9
    assert names[0] == 'Neutral'
    assert names[1] == 'Player2'
    assert client.post('/lobby_join', json={'game_id': '0001', 'real_name': 'PERSON 0'}).status_code == 409
    lobby = server._load_lobby(game)
    lobby['open'] = False
    server._save_lobby(game, lobby)
    assert client.post('/lobby_join', json={'game_id': '0001', 'real_name': 'Late'}).status_code == 409


@pytest.mark.parametrize('endpoint', ['finalize_and_start', 'start_game'])
@pytest.mark.parametrize('count', [0, 1, 2, 4, 5, 6])
def test_start_minimum_without_acknowledgment_gate(server, client, monkeypatch, endpoint, count):
    make_game(server, count=count)
    monkeypatch.setattr(server, 'admin_lobby_state_for_start', lambda *args: True)
    launch = Mock()

    def record_launch(gid, game, **kwargs):
        launch(gid)
        server.running_games[gid] = {'llm': Mock()}

    monkeypatch.setattr(server, '_launch_game_processes', record_launch)
    response = client.post('/api/admin/' + endpoint, json={'game_id': '0001', 'force': True})
    assert response.status_code == (400 if count < 2 else 200)
    assert launch.call_count == (0 if count < 2 else 1)


def test_not_ready_blocks_start_without_closing_lobby(server, client):
    game = make_game(server, count=2)
    assert client.post('/api/admin/finalize_and_start', json={'game_id': '0001'}).status_code == 409
    assert server._load_lobby(game)['open']


def test_ready_requires_event_and_live_process(server, monkeypatch):
    game = make_game(server)
    event = game / 'ready.jsonl'
    monkeypatch.setattr(server, '_game_event_files', lambda *args: [event])
    proc = Mock()
    proc.poll.return_value = None
    server.lobby_ai_procs['0001'] = {'llm': proc}
    assert not server.admin_lobby_state_for_start('0001', game)
    event.write_text('invalid\n' + json.dumps({'event_type': 'agent_ready'}) + '\n', encoding='utf-8')
    assert server.admin_lobby_state_for_start('0001', game)
    proc.poll.return_value = 1
    assert not server.admin_lobby_state_for_start('0001', game)


def test_shared_launcher_minimum_and_missing_config(server):
    game = make_game(server, count=1, opened=False)
    assert server._launch_game_processes('0001', game, force=True)[0] == 400
    (game / server.GAME_CONFIG_FILE).unlink()
    assert server._launch_game_processes('0001', game, force=True)[0] == 400


@pytest.mark.parametrize('game_id', ['game_001', '../0001', '123', '12345'])
def test_dataset_ids_cannot_be_run(server, client, game_id):
    assert not server._valid_game_id(game_id)
    for endpoint in ('start_game', 'finalize_and_start', 'recreate_game'):
        assert client.post('/api/admin/' + endpoint, json={'game_id': game_id}).status_code == 400


@pytest.mark.parametrize('key', ['GEMINI_API_KEY', 'GOOGLE_API_KEY',
                                 'TOGETHER_API_KEY', 'OPENROUTER_API_KEY'])
def test_keys_remain_in_memory_and_are_not_echoed(server, client, key):
    sentinel = 'offline-fake-key-not-a-credential'
    response = client.post('/api/admin/set_api_keys', json={key: sentinel})
    assert response.status_code == 200
    assert sentinel not in response.get_data(as_text=True)
    assert response.get_json()['keys_set'] == [key]
    response = client.get('/api/admin/get_key_status')
    assert response.status_code == 200
    assert sentinel not in response.get_data(as_text=True)
    assert response.get_json()[key] == 'set_ui'
    assert server.server_api_keys.get(key) == sentinel
    assert server._get_env_with_api_keys()[key] == sentinel


def test_no_excluded_api_requests_in_page():
    html = (ROOT / 'templates' / 'game.html').read_text(encoding='utf-8')
    for excluded in ('/prolific_join', '/api/admin/evaluate_game', '/api/admin/eval_status',
                     'viewer_data.json', 'simulation_viewer.html', 'prolific'):
        assert excluded not in html.lower()


def test_web_has_no_private_absolute_paths_or_project_defaults():
    source = (ROOT / 'web_player.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not re.search(r'(?i)(?:^|\s)[a-z]:[\\/][a-z0-9_]|/(?:home|Users)/', node.value), node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (ast.unparse(node.func) == 'os.environ.get' and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == 'GOOGLE_CLOUD_PROJECT'):
                assert len(node.args) == 1 or ast.literal_eval(node.args[1]) in ('', None)


def test_retained_functions_have_no_unresolved_global_names():
    """Check every function, including nested/comprehension scopes, after pruning."""
    source = (ROOT / 'web_player.py').read_text(encoding='utf-8')
    table = symtable.symtable(source, 'web_player.py', 'exec')
    defined = {s.get_name() for s in table.get_symbols()
               if s.is_assigned() or s.is_imported() or s.is_namespace()}
    defined.update(vars(builtins))
    defined.update({'__name__', '__file__', '__package__', '__doc__', '__builtins__'})
    unresolved = []

    def inspect(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in defined:
                unresolved.append((scope.get_name(), scope.get_lineno(), symbol.get_name()))
        for child in scope.get_children():
            inspect(child)

    inspect(table)
    assert not unresolved, unresolved


def test_no_duplicate_top_level_functions():
    tree = ast.parse((ROOT / 'web_player.py').read_text(encoding='utf-8'))
    names = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert len(names) == len(set(names))


def test_page_requests_have_registered_routes(server):
    html = (ROOT / 'templates' / 'game.html').read_text(encoding='utf-8')
    paths = set(re.findall(r"fetch\(\s*['\"`](/[^'\"`?${]+)", html))
    registered = {rule.rule for rule in server.app.url_map.iter_rules()}
    assert paths
    assert not paths - registered, sorted(paths - registered)


@pytest.mark.parametrize('route', ['lobby_state', 'game_status', 'component_state', 'agent_state',
                                  'ai_state', 'ai_metrics', 'ai_issues', 'reviewer_detail',
                                  'votes', 'survey_results'])
def test_retained_admin_read_routes(server, client, monkeypatch, route):
    game = make_game(server, count=2)
    (game / server.LLM_LOG_FILE_FORMAT.format('Player1')).touch()
    events = [
        {'event_type': 'agent_ready'},
        {'event_type': 'snapshot_observed', 'turn_index': 1, 'self_claims': []},
        {'event_type': 'orchestration_mode', 'effective_orchestration_mode': 'synchronous'},
    ]
    server.game_file(game, 'game_events.jsonl', for_write=True).write_text(
        ''.join(json.dumps(event) + '\n' for event in events), encoding='utf-8')
    monkeypatch.setattr(server, '_game_agent_build_info', lambda *args: {'commit': 'offline'})
    response = client.get('/api/admin/' + route + '?game_id=0001')
    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.is_json


def test_lobby_state_and_overview_have_no_roster_cap(server, client, monkeypatch):
    game = make_game(server, count=6)
    monkeypatch.setattr(server, 'iter_all_games', lambda **kwargs: [('test', 'social_turing_test_agent', game)])
    state = client.get('/api/admin/lobby_state?game_id=0001')
    assert state.status_code == 200
    data = state.get_json()
    assert data['joined_count'] == 6
    assert data['max_humans'] is None
    assert data['required_humans'] is None
    assert data['instructions_ack_count'] == 0
    assert data['publication_ready'] is True
    overview = client.get('/api/admin/open_lobbies')
    assert overview.status_code == 200
    assert overview.get_json()['lobbies'][0]['capacity'] is None


def test_recreate_preserves_tunable_settings_without_old_roster_gates(server, client, monkeypatch):
    game = make_game(server, count=6, opened=False)
    config_path = game / server.GAME_CONFIG_FILE
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config.update(max_humans=5, required_humans=5, require_all_instructions_ack=True)
    config['players'][0]['llm_config'].update(num_candidates=6, pipeline_timeout_s=40)
    config_path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(server, '_get_next_game_id', lambda: '0002')
    monkeypatch.setattr(server, 'target_dir_for_config', lambda cfg, gid: server._offline_root / gid)
    response = client.post('/api/admin/recreate_game', json={'game_id': '0001'})
    assert response.status_code == 200, response.get_data(as_text=True)
    new_game = server._offline_root / '0002'
    recreated = json.loads((new_game / server.GAME_CONFIG_FILE).read_text(encoding='utf-8'))
    assert 'max_humans' not in recreated
    assert 'required_humans' not in recreated
    assert recreated['require_all_instructions_ack'] is False
    assert len(recreated['players']) == 1
    assert recreated['players'][0]['llm_config']['num_candidates'] == 6
    assert recreated['players'][0]['llm_config']['pipeline_timeout_s'] == 40
    assert server._load_lobby(new_game)['open']


@pytest.fixture
def local_cli(monkeypatch):
    """Import only config preparation; no Flask startup, SDK clients or input."""
    import importlib

    monkeypatch.syspath_prepend(str(REPOSITORY))
    module = importlib.import_module('social_turing_test.prepare_config')
    assert Path(module.__file__).resolve() == ROOT / 'prepare_config.py'

    def forbidden(*args, **kwargs):
        raise AssertionError('Interactive input, network and processes are forbidden')

    monkeypatch.setattr(builtins, 'input', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    return module


def cli_settings(local_cli, monkeypatch, *arguments):
    monkeypatch.setattr(sys, 'argv', ['prepare_config.py', *arguments])
    args = local_cli.parse_args()
    return args, local_cli.get_llm_config('LLM0', args)


def test_cli_agent_defaults_and_sample_parity(local_cli, monkeypatch):
    args, settings = cli_settings(local_cli, monkeypatch)
    expected = {
        'async_type': 'social_turing_test_agent', 'model_name': 'gemini-2.5-flash',
        'use_gemini': True, 'use_together': False, 'use_openrouter': False,
        'use_pipeline': False, 'scheduler_model': 'gemini-2.5-flash',
        'reviewer_enabled': True, 'reviewer_backend': 'gemini',
        'reviewer_model': 'gemini-2.5-flash', 'multi_candidate_generation': True,
        'num_candidates': 4, 'investigator': True, 'voting_strategy': 'strategic',
        'pipeline_timeout_s': 35.0, 'orchestration_mode': 'synchronous',
        'allow_active_authoritative': False, 'persona': 'swiss_student',
        'agent_code': 'current',
    }
    assert {key: settings[key] for key in expected} == expected
    from social_turing_test.core.game_constants import DEFAULT_GAME_CONFIG
    config = json.loads((ROOT / DEFAULT_GAME_CONFIG).read_text(encoding='utf-8'))
    humans = [p for p in config['players'] if not p['is_llm']]
    ais = [p for p in config['players'] if p['is_llm']]
    assert len(humans) == 2 and len(ais) == 1
    assert ais[0]['is_ai'] is True
    assert ais[0]['llm_config'] == settings
    assert_current_config(config)
    assert_current_config(settings)
    assert 'gemini-2.5-pro' not in json.dumps(config)
    assert 'gemini-2.5-pro' not in json.dumps(settings)
    assert len(local_cli.handle_num_players(args)) == 3


@pytest.mark.parametrize('model,provider', [
    ('gemini-2.5-pro', 'gemini'),
    ('Qwen/Qwen3-Next-80B-A3B-Instruct', 'together'),
    ('openai/gpt-4o-mini', 'openrouter'),
])
def test_cli_explicit_models_keep_provider_and_agent(local_cli, monkeypatch, model, provider):
    _, settings = cli_settings(local_cli, monkeypatch,
                               '--generator-model', model,
                               '--scheduler-model', 'gemini-2.5-pro')
    assert settings['model_name'] == model
    assert settings['scheduler_model'] == 'gemini-2.5-pro'
    for candidate in ('gemini', 'together', 'openrouter'):
        assert settings['use_' + candidate] is (candidate == provider)
    assert settings['async_type'] == 'social_turing_test_agent'


@pytest.mark.parametrize('provider', ['gemini', 'together', 'openrouter', 'huggingface'])
def test_cli_explicit_provider_wins_over_detection(local_cli, monkeypatch, provider):
    _, settings = cli_settings(local_cli, monkeypatch,
                               '--generator-model', 'custom/generator', '--provider', provider)
    assert settings['model_name'] == 'custom/generator'
    assert settings['use_pipeline'] is (provider == 'huggingface')
    for candidate in ('gemini', 'together', 'openrouter'):
        assert settings['use_' + candidate] is (candidate == provider)
    assert settings['async_type'] == 'social_turing_test_agent'


@pytest.mark.parametrize('mode', ['none', 'gemini', 'together', 'openrouter'])
def test_cli_reviewer_selection_never_restores_old_async(local_cli, monkeypatch, mode):
    _, settings = cli_settings(local_cli, monkeypatch, '--review-mode', mode,
                               '--reviewer-model', 'custom/reviewer')
    assert settings['async_type'] == 'social_turing_test_agent'
    assert settings['reviewer_enabled'] is (mode != 'none')
    if mode == 'none':
        assert 'reviewer_backend' not in settings
        assert 'reviewer_model' not in settings
    else:
        assert settings['reviewer_backend'] == mode
        assert settings['reviewer_model'] == 'custom/reviewer'


def test_cli_json_choices_survive_defaults_and_cli_overrides(local_cli, monkeypatch, tmp_path):
    supplied = {
        'model_name': 'custom/generator', 'use_gemini': False, 'use_together': False,
        'use_openrouter': True, 'reviewer_backend': 'openrouter',
        'reviewer_model': 'custom/reviewer', 'reviewer_enabled': False,
        'scheduler_model': 'custom/scheduler', 'num_candidates': 6,
        'pipeline_timeout_s': 40, 'persona': 'american_student',
        'async_type': 'reviewed_schedule_then_generate',
        'orchestration_mode': 'active',
    }
    path = tmp_path / 'operator.json'
    path.write_text(json.dumps(supplied), encoding='utf-8')
    arguments = ('--llm_config_json_path', str(path))
    _, settings = cli_settings(local_cli, monkeypatch, *arguments)
    for key, value in supplied.items():
        if key not in ('async_type', 'orchestration_mode'):
            assert settings[key] == value
    assert settings['async_type'] == 'social_turing_test_agent'
    assert settings['orchestration_mode'] == 'synchronous'
    assert_current_config(settings)
    _, settings = cli_settings(local_cli, monkeypatch, *arguments,
                               '--generator-model', 'gemini-2.5-pro',
                               '--review-mode', 'together', '--reviewer-model', 'custom/override')
    assert settings['model_name'] == 'gemini-2.5-pro'
    assert settings['use_gemini'] is True and settings['use_openrouter'] is False
    assert settings['reviewer_enabled'] is True
    assert settings['reviewer_backend'] == 'together'
    assert settings['reviewer_model'] == 'custom/override'


def test_cli_name_fallback_only_after_pool_exhaustion(local_cli, monkeypatch):
    monkeypatch.setattr(local_cli, 'OPTIONAL_CODE_NAMES', ['Player1', 'Neutral'])
    args, _ = cli_settings(local_cli, monkeypatch, '--players', '2')
    with pytest.raises(ValueError, match='Minimum is 3'):
        local_cli.handle_num_players(args)
    args.players = 2 + 60
    players = local_cli.handle_num_players(args)
    names = [p.name for p in players]
    assert len(names) == len(set(names)) == 62
    assert set(names[:2]) == {'Player1', 'Neutral'}
    assert names[2] == 'Player2'
    assert sum(p.is_ai for p in players) == 1
    args.players = 3
    monkeypatch.setattr(local_cli, 'OPTIONAL_CODE_NAMES', ['Alex', 'Blake', 'Casey', 'Drew'])
    assert {p.name for p in local_cli.handle_num_players(args)} <= set(local_cli.OPTIONAL_CODE_NAMES)


def test_local_sample_prepares_agent_without_touching_dataset(local_cli, monkeypatch, tmp_path):
    from social_turing_test import prepare_game
    from social_turing_test.core import game_paths, game_identity
    from social_turing_test.core.game_constants import DEFAULT_GAME_CONFIG

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(game_paths, 'DIRS_PREFIX', str(tmp_path / 'games'))
    published = tmp_path / 'games' / 'game_001'
    published.mkdir(parents=True)
    receipt = published / 'metadata.json'
    receipt.write_text('{"published": true}', encoding='utf-8')
    before = receipt.read_bytes()
    prepare_game.init_game('0001', str(ROOT / DEFAULT_GAME_CONFIG))
    game = game_paths.resolve_game_dir('0001')
    assert game == tmp_path / 'games' / 'real' / '0001'
    config = json.loads((game / 'config.json').read_text(encoding='utf-8'))
    assert game_identity.CURRENT_AGENT_IDENTITY == 'social_turing_test_agent'
    assert config['players'][-1]['llm_config'][game_identity.AGENT_IDENTITY_KEY] == 'social_turing_test_agent'
    assert game_paths.next_game_id() == '0002'
    assert receipt.read_bytes() == before
    assert game_paths.target_dir_for_config({'is_test': True}, '0002') == (
        tmp_path / 'games' / 'test' / '0002')
    for invalid in ('game_001', '../0001', '123', '12345', '0001\n'):
        with pytest.raises(ValueError, match='Invalid game_id'):
            game_paths.resolve_game_dir(invalid)
        with pytest.raises(ValueError, match='Invalid game_id'):
            game_paths.target_dir_for_config(config, invalid)
    assert [gd.name for _, _, gd in game_paths.iter_all_games()] == ['0001']


def test_local_runtime_paths_resolve_real_and_test(local_cli, monkeypatch, tmp_path):
    from social_turing_test.core import game_paths

    monkeypatch.setattr(game_paths, 'DIRS_PREFIX', str(tmp_path))
    for relative, gid in [('real/0001', '0001'), ('test/0002', '0002'),
                          ('real/0003', '0003')]:
        game = tmp_path / relative
        game.mkdir(parents=True)
        (game / 'config.json').write_text('{"players": []}', encoding='utf-8')
        assert game_paths.resolve_game_dir(gid) == game
    assert game_paths.resolve_game_dir('0004') == tmp_path / 'real' / '0004'
    assert game_paths.next_game_id() == '0004'


def test_cli_provider_only_override_and_test_routing(local_cli, monkeypatch, tmp_path):
    from social_turing_test.core import game_paths

    args, settings = cli_settings(local_cli, monkeypatch, '--provider', 'openrouter', '--test')
    assert settings['model_name'] == 'gemini-2.5-flash'
    assert settings['use_openrouter'] is True
    assert settings['use_gemini'] is False and settings['use_together'] is False
    assert settings['reviewer_backend'] == 'gemini'
    monkeypatch.setattr(game_paths, 'DIRS_PREFIX', str(tmp_path))
    config = {'is_test': args.test, 'players': [
        {'is_ai': True, 'is_llm': True, 'llm_config': settings},
    ]}
    assert game_paths.target_dir_for_config(config, '0001') == tmp_path / 'test' / '0001'


@pytest.mark.parametrize('key', RETIRED_KEYS)
@pytest.mark.parametrize('value', [False, None, 'offline-retired-value'])
@pytest.mark.parametrize('placement', ['direct', 'dict', 'list'])
def test_retired_config_helper_rejects_recursively_without_mutation(key, value, placement):
    from social_turing_test.core.config_schema import ConfigError, reject_retired_config

    payload = {key: value}
    if placement == 'dict':
        payload = {'outer': {'settings': payload}}
    elif placement == 'list':
        payload = {'outer': [None, {'settings': [payload]}]}
    before = json.dumps(payload)
    assert issubclass(ConfigError, ValueError)
    with pytest.raises(ConfigError, match=key) as error:
        reject_retired_config(payload)
    assert 'retired' in str(error.value).lower()
    assert 'offline-retired-value' not in str(error.value)
    assert json.dumps(payload) == before


@pytest.mark.parametrize('key', RETIRED_KEYS)
@pytest.mark.parametrize('value', [False, None, 'offline-retired-value'])
@pytest.mark.parametrize('placement', ['top_level', 'advanced', 'nested_api'])
def test_web_rejects_retired_settings_before_saving(server, client, key, value, placement):
    payload = {key: value}
    if placement == 'advanced':
        payload = {'advanced_llm_config': payload}
    elif placement == 'nested_api':
        payload = {'advanced_llm_config': {'custom': [{'settings': payload}]}}
    response = client.post('/api/admin/create_config', json=payload)
    assert response.status_code == 400, response.get_data(as_text=True)
    assert key in response.get_json()['error']
    assert not (server._offline_root / 'configurations').exists()


@pytest.mark.parametrize('key', RETIRED_KEYS)
@pytest.mark.parametrize('value', [False, None, 'offline-retired-value'])
@pytest.mark.parametrize('nested', [False, True])
def test_cli_json_rejects_retired_settings(local_cli, monkeypatch, tmp_path, key, value, nested):
    from social_turing_test.core.config_schema import ConfigError

    payload = {key: value}
    if nested:
        payload = {'custom': [{'settings': payload}]}
    path = tmp_path / 'retired.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ConfigError, match=key):
        cli_settings(local_cli, monkeypatch, '--llm_config_json_path', str(path))
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['retired.json']


def inject_retired_config(config, key, value, placement):
    if placement == 'top_level':
        config[key] = value
    elif placement == 'settings':
        config['players'][0]['llm_config'][key] = value
    else:
        config['players'][0]['llm_config']['custom'] = [{'settings': {key: value}}]


@pytest.mark.parametrize('key', RETIRED_KEYS)
@pytest.mark.parametrize('value', [False, None, 'offline-retired-value'])
@pytest.mark.parametrize('placement', ['top_level', 'settings', 'nested'])
def test_saved_config_and_recreation_reject_retired_settings_before_write(
        server, client, monkeypatch, key, value, placement):
    game = make_game(server, count=2, opened=False)
    path = game / server.GAME_CONFIG_FILE
    config = json.loads(path.read_text(encoding='utf-8'))
    inject_retired_config(config, key, value, placement)
    path.write_text(json.dumps(config), encoding='utf-8')
    before = {p.relative_to(server._offline_root): p.read_bytes()
              for p in server._offline_root.rglob('*') if p.is_file()}
    monkeypatch.setattr(server, '_get_next_game_id', lambda: '0002')
    route_target = Mock(side_effect=AssertionError('Rejected config must not be routed'))
    launch = Mock(side_effect=AssertionError('Rejected config must not launch an AI'))
    cleanup = Mock(side_effect=AssertionError('Rejected recreation must not stop the old AI'))
    monkeypatch.setattr(server, 'target_dir_for_config', route_target)
    monkeypatch.setattr(server, '_launch_lobby_ai', launch)
    monkeypatch.setattr(server, '_cleanup_lobby_ai', cleanup)
    assert key in server._local_config_error(config)
    for endpoint, payload in [
        ('create_game', {'config_path': str(path)}),
        ('recreate_game', {'game_id': '0001'}),
    ]:
        response = client.post('/api/admin/' + endpoint, json=payload)
        assert response.status_code == 400, response.get_data(as_text=True)
        assert key in response.get_json()['error']
    assert {p.relative_to(server._offline_root): p.read_bytes()
            for p in server._offline_root.rglob('*') if p.is_file()} == before
    route_target.assert_not_called()
    launch.assert_not_called()
    cleanup.assert_not_called()


@pytest.mark.parametrize('key', RETIRED_KEYS)
@pytest.mark.parametrize('value', [False, None, 'offline-retired-value'])
@pytest.mark.parametrize('placement', ['top_level', 'settings', 'nested'])
def test_prepare_game_rejects_retired_settings_before_any_write(
        local_cli, monkeypatch, tmp_path, key, value, placement):
    from social_turing_test import prepare_game
    from social_turing_test.core import game_paths
    from social_turing_test.core.config_schema import ConfigError
    from social_turing_test.core.game_constants import DEFAULT_GAME_CONFIG

    config = json.loads((ROOT / DEFAULT_GAME_CONFIG).read_text(encoding='utf-8'))
    # Put the AI first so this fixture exercises the same placements as the web API.
    config['players'].sort(key=lambda p: not p.get('is_llm'))
    inject_retired_config(config, key, value, placement)
    path = tmp_path / 'retired-game.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    before = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(game_paths, 'DIRS_PREFIX', str(tmp_path / 'games'))
    route_target = Mock(side_effect=AssertionError('Rejected config must not be routed'))
    monkeypatch.setattr(game_paths, 'target_dir_for_config', route_target)
    with pytest.raises(ConfigError, match=key):
        prepare_game.init_game('0001', str(path))
    route_target.assert_not_called()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['retired-game.json']


def test_current_ui_and_schema_have_no_evaluator_controls(client):
    from social_turing_test.core.config_schema import CONFIG_SCHEMA, apply_defaults

    html = (ROOT / 'templates' / 'game.html').read_text(encoding='utf-8')
    for retired in ('cfgLiveEval', 'live_evaluator', 'Live Evaluator', 'liveEvaluator'):
        assert retired.lower() not in html.lower()
    assert_current_config({item['key']: item.get('default') for item in CONFIG_SCHEMA})
    assert_current_config(apply_defaults({'async_type': 'social_turing_test_agent'}))
    response = client.get('/')
    assert response.status_code == 200
    assert b'cfgLiveEval' not in response.data


def test_activity_ignores_retired_events_and_preserves_six_retained_workers(server):
    events = [
        {'event_type': 'live_evaluator_tick', 'timestamp': 'offline'},
        {'event_type': 'live_evaluator_phase1', 'parsed': {'suspect': 'Player1'}},
        {'event_type': 'live_evaluator_phase2', 'self_weaknesses': ['retired']},
    ]
    activity = server._agent_subagents_activity(events)
    assert activity == server._agent_subagents_activity([])
    assert 'live_evaluator' not in json.dumps(activity)
    workers = {'internet_context', 'round_summarizer', 'persona_monitor',
               'social_state_tracker', 'behavior_adapter', 'repetition_monitor'}
    assert workers <= activity.keys()
    assert {key for key, value in activity.items()
            if isinstance(value, dict) and value.get('enabled', True)} == workers
    assert 'player_profiler' not in activity


def test_runtime_worker_declaration_contains_only_six_retained_workers():
    """Inspect the real constructor without instantiating paid SDKs or threads.

    Factory construction/request spies are covered by the credentials test module.
    """
    path = ROOT / 'llm_players' / 'agent' / 'agent.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    declarations = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Attribute)
                    and ast.unparse(node.target) == 'self.sub_agents']
    assert len(declarations) == 1
    assert isinstance(declarations[0], ast.List)
    assert [ast.unparse(node.func) for node in declarations[0].elts] == [
        'InternetContextAgent', 'RoundSummarizerAgent', 'PersonaMonitorAgent',
        'SocialStateTrackerAgent', 'BehaviorAdapterAgent', 'RepetitionMonitorAgent',
    ]