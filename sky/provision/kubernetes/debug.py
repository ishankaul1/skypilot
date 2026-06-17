"""Best-effort collection of a cluster's Kubernetes resources for debug dumps.

When the API server dumps a cluster that runs on Kubernetes, we also snapshot
the related k8s objects -- the pods, their events, the Services/Deployments/etc.
SkyPilot creates during instance creation, and (if Kueue is in use) the gang
Workload -- so an operator can inspect the cluster the way they would with
``kubectl get -o yaml``, without needing kubectl access to the user's cluster.

This module is intentionally provider-specific and self-contained: the generic
dump path (``sky.utils.debug_utils``) discovers a cluster's k8s coordinates from
its command runners and delegates the actual API queries here. Every query is
best-effort -- a failure is recorded and returned, never raised, so one missing
or inaccessible object can't abort the surrounding dump.
"""
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.adaptors import kubernetes
from sky.provision import constants as provision_constants
from sky.utils import yaml_utils

logger = sky_logging.init_logger(__name__)

# Kueue stamps these labels on the pods it manages (see
# templates/kubernetes-ray.yml.j2). It creates one Workload per pod-group, and
# SkyPilot sets the group name to the cluster's on-cloud name, so we find the
# Workload(s) for a cluster by selecting on this label.
_KUEUE_POD_GROUP_LABEL = 'kueue.x-k8s.io/pod-group-name'
# Kueue Workload CRD coordinates for the custom objects API.
_KUEUE_GROUP = 'kueue.x-k8s.io'
_KUEUE_VERSION = 'v1beta1'
_KUEUE_WORKLOADS_PLURAL = 'workloads'

# Keys in a serialized Secret whose values must never leave the cluster.
_SECRET_REDACTED_KEYS = ('data', 'stringData')
_REDACTED = '<redacted>'


def dump_cluster_resources(context: Optional[str], namespace: str,
                           cluster_name_on_cloud: str, pod_names: List[str],
                           output_dir: str) -> List[Dict[str, str]]:
    """Snapshot a Kubernetes cluster's related resources into ``output_dir``.

    Writes (best-effort, skipping anything that doesn't exist)::

        <output_dir>/
          pods/<pod>.yaml      # full pod spec + status
          events/<pod>.yaml    # events involving each pod
          services.yaml        # resources carrying the skypilot-cluster-name
          deployments.yaml     #   label -- i.e. what SkyPilot created during
          ...                  #   instance creation
          workloads.yaml       # Kueue Workload(s), if Kueue is in use

    Args:
        context: kube context the cluster lives in (None = current context).
        namespace: namespace the cluster's resources live in.
        cluster_name_on_cloud: value of the ``skypilot-cluster-name`` label,
            used to select the resources SkyPilot created for this cluster.
        pod_names: the cluster's pod names (head first), from its runners.
        output_dir: directory to write the resource files into.

    Returns:
        A list of error records, each ``{'resource', 'error', 'traceback'}``,
        for queries that failed. ``resource`` is dump-relative (e.g.
        ``kubernetes/pods/<pod>``) so the caller can prefix it with the cluster
        name. Empty if everything succeeded (or simply didn't exist).
    """
    errors: List[Dict[str, str]] = []
    os.makedirs(output_dir, exist_ok=True)

    # The ApiClient turns typed k8s model objects into kubectl-style dicts
    # (camelCase keys), so the dumped YAML matches `kubectl get -o yaml`.
    api_client = kubernetes.api_client(context)

    def to_dict(obj: Any) -> Dict[str, Any]:
        return api_client.sanitize_for_serialization(obj)

    _dump_pods(context, namespace, pod_names, output_dir, to_dict, errors)
    _dump_labeled_resources(context, namespace, cluster_name_on_cloud,
                            output_dir, to_dict, errors)
    _dump_kueue_workloads(context, namespace, cluster_name_on_cloud, output_dir,
                          errors)
    return errors


def _record_error(errors: List[Dict[str, str]], resource: str,
                  e: Exception) -> None:
    logger.debug(f'Failed to collect {resource}: {e}')
    errors.append({
        'resource': resource,
        'error': str(e),
        'traceback': traceback.format_exc(),
    })


def _write_yaml(path: str, obj: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(yaml_utils.dump_yaml_str(obj))


def _dump_pods(context: Optional[str], namespace: str, pod_names: List[str],
               output_dir: str, to_dict: Callable[[Any], Dict[str, Any]],
               errors: List[Dict[str, str]]) -> None:
    """Dump each pod's spec/status and the events involving it."""
    if not pod_names:
        return
    core = kubernetes.core_api(context)
    pods_dir = os.path.join(output_dir, 'pods')
    events_dir = os.path.join(output_dir, 'events')
    os.makedirs(pods_dir, exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)

    for pod_name in pod_names:
        try:
            pod = core.read_namespaced_pod(
                pod_name, namespace, _request_timeout=kubernetes.API_TIMEOUT)
            _write_yaml(os.path.join(pods_dir, f'{pod_name}.yaml'),
                        to_dict(pod))
        except Exception as e:  # pylint: disable=broad-except
            _record_error(errors, f'kubernetes/pods/{pod_name}', e)

        # Events are objects in the namespace that reference the pod; select
        # them with a field selector on involvedObject (mirrors
        # instance._get_pod_events).
        try:
            field_selector = (f'involvedObject.kind=Pod,'
                              f'involvedObject.name={pod_name}')
            events = core.list_namespaced_event(
                namespace,
                field_selector=field_selector,
                _request_timeout=kubernetes.API_TIMEOUT).items
            if events:
                _write_yaml(os.path.join(events_dir, f'{pod_name}.yaml'),
                            [to_dict(ev) for ev in events])
        except Exception as e:  # pylint: disable=broad-except
            _record_error(errors, f'kubernetes/events/{pod_name}', e)


def _dump_labeled_resources(context: Optional[str], namespace: str,
                            cluster_name_on_cloud: str, output_dir: str,
                            to_dict: Callable[[Any], Dict[str, Any]],
                            errors: List[Dict[str, str]]) -> None:
    """Dump the resources SkyPilot created for this cluster.

    Everything SkyPilot creates during instance creation carries the
    ``skypilot-cluster-name`` label (it's how teardown finds them), so a single
    label selector covers Services and any other kinds we create now or later.
    """
    label_selector = (f'{provision_constants.TAG_SKYPILOT_CLUSTER_NAME}='
                      f'{cluster_name_on_cloud}')
    core = kubernetes.core_api(context)

    # (filename, lister) for the kinds SkyPilot may create. Listing a kind we
    # didn't create just returns an empty list, so an over-broad sweep is safe
    # and future-proofs us against new resource kinds.
    kinds: List[Tuple[str, Callable[[], List[Any]]]] = [
        ('services.yaml', lambda: core.list_namespaced_service(
            namespace,
            label_selector=label_selector,
            _request_timeout=kubernetes.API_TIMEOUT).items),
        ('deployments.yaml',
         lambda: kubernetes.apps_api(context).list_namespaced_deployment(
             namespace,
             label_selector=label_selector,
             _request_timeout=kubernetes.API_TIMEOUT).items),
        ('persistent_volume_claims.yaml',
         lambda: core.list_namespaced_persistent_volume_claim(
             namespace,
             label_selector=label_selector,
             _request_timeout=kubernetes.API_TIMEOUT).items),
        ('config_maps.yaml', lambda: core.list_namespaced_config_map(
            namespace,
            label_selector=label_selector,
            _request_timeout=kubernetes.API_TIMEOUT).items),
        ('ingresses.yaml',
         lambda: kubernetes.networking_api(context).list_namespaced_ingress(
             namespace,
             label_selector=label_selector,
             _request_timeout=kubernetes.API_TIMEOUT).items),
    ]
    for filename, lister in kinds:
        try:
            items = lister()
            if not items:
                continue
            _write_yaml(os.path.join(output_dir, filename),
                        [to_dict(item) for item in items])
        except Exception as e:  # pylint: disable=broad-except
            _record_error(errors, f'kubernetes/{filename}', e)

    # Secrets are dumped separately so their values can be redacted -- we want
    # the metadata (which secrets exist, their keys) for debugging, never the
    # contents.
    try:
        secrets = core.list_namespaced_secret(
            namespace,
            label_selector=label_selector,
            _request_timeout=kubernetes.API_TIMEOUT).items
        if secrets:
            redacted = []
            for secret in secrets:
                secret_dict = to_dict(secret)
                for key in _SECRET_REDACTED_KEYS:
                    values = secret_dict.get(key)
                    if values:
                        secret_dict[key] = {k: _REDACTED for k in values}
                redacted.append(secret_dict)
            _write_yaml(os.path.join(output_dir, 'secrets.yaml'), redacted)
    except Exception as e:  # pylint: disable=broad-except
        _record_error(errors, 'kubernetes/secrets', e)


def _dump_kueue_workloads(context: Optional[str], namespace: str,
                          cluster_name_on_cloud: str, output_dir: str,
                          errors: List[Dict[str, str]]) -> None:
    """Dump the Kueue Workload(s) for this cluster, if Kueue is in use.

    The custom objects API returns plain dicts (not typed models), so no
    serialization is needed. If the Kueue CRD isn't installed the API returns
    404 -- that just means the cluster doesn't use Kueue, so we skip silently.
    """
    label_selector = f'{_KUEUE_POD_GROUP_LABEL}={cluster_name_on_cloud}'
    try:
        response = kubernetes.custom_objects_api(
            context).list_namespaced_custom_object(
                group=_KUEUE_GROUP,
                version=_KUEUE_VERSION,
                namespace=namespace,
                plural=_KUEUE_WORKLOADS_PLURAL,
                label_selector=label_selector,
                _request_timeout=kubernetes.API_TIMEOUT)
    except kubernetes.api_exception() as e:
        if e.status == 404:
            logger.debug('Kueue Workload CRD not found; cluster is not using '
                         'Kueue, skipping.')
            return
        _record_error(errors, 'kubernetes/workloads', e)
        return
    except Exception as e:  # pylint: disable=broad-except
        _record_error(errors, 'kubernetes/workloads', e)
        return

    items = response.get('items', []) if isinstance(response, dict) else []
    if items:
        _write_yaml(os.path.join(output_dir, 'workloads.yaml'), items)
