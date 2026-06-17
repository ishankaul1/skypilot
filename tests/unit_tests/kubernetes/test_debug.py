"""Tests for sky.provision.kubernetes.debug (cluster resource dump)."""
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from sky.provision.kubernetes import debug
import sky.utils.yaml_utils as yaml_utils


class _FakeApiException(Exception):
    """Stand-in for kubernetes.client.rest.ApiException."""

    def __init__(self, status):
        self.status = status
        super().__init__(f'status={status}')


def _obj(marker):
    """A fake k8s model object; sanitize_for_serialization keys off _marker."""
    return SimpleNamespace(_marker=marker)


def _list(*markers):
    return SimpleNamespace(items=[_obj(m) for m in markers])


@pytest.fixture
def k8s_apis(monkeypatch):
    """Patch every kubernetes adaptor call the module makes.

    Returns the individual API mocks so each test configures its own return
    values; defaults are empty lists / a happy-path pod read.
    """
    core = mock.MagicMock()
    core.read_namespaced_pod.return_value = _obj('pod')
    core.list_namespaced_event.return_value = _list()
    core.list_namespaced_service.return_value = _list()
    core.list_namespaced_persistent_volume_claim.return_value = _list()
    core.list_namespaced_config_map.return_value = _list()
    core.list_namespaced_secret.return_value = _list()

    apps = mock.MagicMock()
    apps.list_namespaced_deployment.return_value = _list()

    networking = mock.MagicMock()
    networking.list_namespaced_ingress.return_value = _list()

    custom = mock.MagicMock()
    custom.list_namespaced_custom_object.return_value = {'items': []}

    api_client = mock.MagicMock()
    # kubectl-style serialization: just surface the object's marker.
    api_client.sanitize_for_serialization.side_effect = (lambda obj: {
        'marker': obj._marker
    })

    monkeypatch.setattr('sky.adaptors.kubernetes.core_api',
                        lambda *a, **k: core)
    monkeypatch.setattr('sky.adaptors.kubernetes.apps_api',
                        lambda *a, **k: apps)
    monkeypatch.setattr('sky.adaptors.kubernetes.networking_api',
                        lambda *a, **k: networking)
    monkeypatch.setattr('sky.adaptors.kubernetes.custom_objects_api',
                        lambda *a, **k: custom)
    monkeypatch.setattr('sky.adaptors.kubernetes.api_client',
                        lambda *a, **k: api_client)
    monkeypatch.setattr('sky.adaptors.kubernetes.api_exception',
                        lambda: _FakeApiException)

    return SimpleNamespace(core=core,
                           apps=apps,
                           networking=networking,
                           custom=custom,
                           api_client=api_client)


def _run(tmp_path, pod_names=('head', 'worker')):
    return debug.dump_cluster_resources(context='ctx',
                                        namespace='ns',
                                        cluster_name_on_cloud='cluster-abc',
                                        pod_names=list(pod_names),
                                        output_dir=str(tmp_path))


def test_dumps_pod_spec_for_each_pod(tmp_path, k8s_apis):
    errors = _run(tmp_path)

    assert not errors
    for pod in ('head', 'worker'):
        path = tmp_path / 'pods' / f'{pod}.yaml'
        assert path.exists()
        assert yaml_utils.read_yaml(str(path)) == {'marker': 'pod'}


def test_events_written_only_when_present(tmp_path, k8s_apis):
    # head has an event, worker has none.
    def events(_namespace, field_selector, **_kwargs):
        if 'head' in field_selector:
            return _list('evt')
        return _list()

    k8s_apis.core.list_namespaced_event.side_effect = events

    _run(tmp_path)

    assert (tmp_path / 'events' / 'head.yaml').exists()
    assert not (tmp_path / 'events' / 'worker.yaml').exists()


def test_labeled_resources_use_cluster_label(tmp_path, k8s_apis):
    k8s_apis.core.list_namespaced_service.return_value = _list('svc')

    _run(tmp_path)

    assert (tmp_path / 'services.yaml').exists()
    _, kwargs = k8s_apis.core.list_namespaced_service.call_args
    assert kwargs['label_selector'] == 'skypilot-cluster-name=cluster-abc'


def test_empty_resource_kinds_are_skipped(tmp_path, k8s_apis):
    # All listers default to empty -> no resource files (only the pods dir).
    _run(tmp_path)

    for name in ('services.yaml', 'deployments.yaml', 'config_maps.yaml',
                 'persistent_volume_claims.yaml', 'ingresses.yaml',
                 'secrets.yaml', 'workloads.yaml'):
        assert not (tmp_path / name).exists()


def test_secret_values_are_redacted(tmp_path, k8s_apis):
    k8s_apis.core.list_namespaced_secret.return_value = _list('secret')
    # The real sanitize_for_serialization would surface the secret's data and
    # stringData; emulate that so the redaction path has something to scrub.
    k8s_apis.api_client.sanitize_for_serialization.side_effect = lambda obj: {
        'metadata': {
            'name': 's'
        },
        'data': {
            'token': 'c2VjcmV0',
            'password': 'aHVudGVyMg=='
        },
        'stringData': {
            'note': 'plaintext'
        },
    }

    errors = _run(tmp_path, pod_names=[])

    assert not errors
    dumped = yaml_utils.read_yaml(str(tmp_path / 'secrets.yaml'))
    assert dumped['data'] == {'token': '<redacted>', 'password': '<redacted>'}
    assert dumped['stringData'] == {'note': '<redacted>'}


def test_kueue_workloads_dumped_when_present(tmp_path, k8s_apis):
    k8s_apis.custom.list_namespaced_custom_object.return_value = {
        'items': [{
            'metadata': {
                'name': 'wl'
            }
        }]
    }

    errors = _run(tmp_path)

    assert not errors
    path = tmp_path / 'workloads.yaml'
    assert path.exists()
    assert yaml_utils.read_yaml(str(path)) == {'metadata': {'name': 'wl'}}
    _, kwargs = k8s_apis.custom.list_namespaced_custom_object.call_args
    assert kwargs['group'] == 'kueue.x-k8s.io'
    assert kwargs['plural'] == 'workloads'
    assert kwargs['label_selector'] == (
        'kueue.x-k8s.io/pod-group-name=cluster-abc')


def test_kueue_crd_absent_is_not_an_error(tmp_path, k8s_apis):
    k8s_apis.custom.list_namespaced_custom_object.side_effect = (
        _FakeApiException(status=404))

    errors = _run(tmp_path)

    assert not errors
    assert not (tmp_path / 'workloads.yaml').exists()


def test_kueue_non_404_failure_is_recorded(tmp_path, k8s_apis):
    k8s_apis.custom.list_namespaced_custom_object.side_effect = (
        _FakeApiException(status=500))

    errors = _run(tmp_path)

    assert len(errors) == 1
    assert errors[0]['resource'] == 'kubernetes/workloads'


def test_pod_read_failure_is_recorded(tmp_path, k8s_apis):
    k8s_apis.core.read_namespaced_pod.side_effect = RuntimeError('boom')

    errors = _run(tmp_path, pod_names=['head'])

    # One error per pod; the dump still continues for everything else.
    assert len(errors) == 1
    assert errors[0]['resource'] == 'kubernetes/pods/head'
    assert errors[0]['error'] == 'boom'
    assert 'traceback' in errors[0]
