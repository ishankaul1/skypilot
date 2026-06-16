"""Scratch benchmark: serialization pulled DIRECTLY off real FastAPI routes.

Two real routes are registered (differing only in response_class), then we pull
each route's response_field + response_class off the route object and replicate
exactly what fastapi/routing.py::get_request_handler does -- no HTTP, no
TestClient overhead, no approximation.

The dispatch we mirror (from get_request_handler / serialize_response):

  use_dump_json = response_field is not None and isinstance(
      response_class, DefaultPlaceholder)        # i.e. NO custom response_class
  serializer = field.serialize_json if dump_json else field.serialize

  /default  -> response_class is DefaultPlaceholder -> use_dump_json=True
               field.serialize_json(value) -> JSON bytes via pydantic-core (one pass)
  /orjson   -> response_class=ORJSONResponse        -> use_dump_json=False
               field.serialize(value) -> dict, then ORJSONResponse(content).body
               (real .render() w/ real orjson options; also pays the @deprecated
                wrapper cost on every instantiation, exactly like prod did)

Both routes share the SAME response_field (built from `-> RequestPayload`), so
the only thing that differs is the path selected by each route's response_class.

Run in the env where sky is installed:
    python bench_routes_serialization.py

NOT for commit -- delete before pushing.
"""
import gc
import statistics
import time
import warnings

import fastapi
from fastapi import responses as fastapi_responses
from fastapi.datastructures import DefaultPlaceholder
import orjson

from sky.server.requests import payloads

# Instantiating ORJSONResponse trips its @deprecated warning every call; silence
# it so it doesn't spam the timed loop (the cost of the warn() call itself is
# still incurred on the orjson arm -- which is realistic).
warnings.simplefilter('ignore')

app = fastapi.FastAPI()


@app.get('/orjson', response_class=fastapi_responses.ORJSONResponse)
async def ep_orjson() -> payloads.RequestPayload:  # body irrelevant; never called
    raise NotImplementedError


@app.get('/default')
async def ep_default() -> payloads.RequestPayload:
    raise NotImplementedError


def _find_route(path: str) -> fastapi.routing.APIRoute:
    for r in app.routes:
        if isinstance(r, fastapi.routing.APIRoute) and r.path == path:
            return r
    raise AssertionError(f'route {path} not found')


_ROUTE_ORJSON = _find_route('/orjson')
_ROUTE_DEFAULT = _find_route('/default')


def validate(route: fastapi.routing.APIRoute, p: payloads.RequestPayload):
    """Validate pass that precedes serialization in serialize_response (common
    to both arms; run once, not timed)."""
    value, errors = route.response_field.validate(p, {}, loc=('response',))
    assert not errors, errors
    return value


def serialize_via_route(route: fastapi.routing.APIRoute, value) -> bytes:
    """Replicate get_request_handler's serialization dispatch for `route`."""
    field = route.response_field
    rc = route.response_class
    use_dump_json = field is not None and isinstance(rc, DefaultPlaceholder)
    kwargs = dict(
        include=route.response_model_include,
        exclude=route.response_model_exclude,
        by_alias=route.response_model_by_alias,
        exclude_unset=route.response_model_exclude_unset,
        exclude_defaults=route.response_model_exclude_defaults,
        exclude_none=route.response_model_exclude_none,
    )
    if use_dump_json:
        return field.serialize_json(value, **kwargs)  # default fast path
    content = field.serialize(value, **kwargs)  # ORJSONResponse path: dict
    actual_rc = rc.value if isinstance(rc, DefaultPlaceholder) else rc
    return actual_rc(content).body  # real ORJSONResponse.render() via orjson


def make_payload(n_clusters: int) -> payloads.RequestPayload:
    clusters = [{
        'name': f'cluster-{i}',
        'launched_at': 1700000000 + i,
        'status': 'UP',
        'resources_str': '1x AWS(m6i.2xlarge, {V100:1})',
        'region': 'us-east-1',
        'autostop': -1,
        'to_down': False,
        'metadata': {
            'tags': ['exp', 'gpu'],
            'cost': 1.234 + i
        },
    } for i in range(n_clusters)]
    return_value = orjson.dumps(clusters).decode('utf-8')
    return payloads.RequestPayload(
        request_id='abcdef01-2345-6789-abcd-ef0123456789',
        name='status',
        entrypoint='gASV...(pickled-base64)...',
        request_body='gASV...(pickled-base64)...',
        status='SUCCEEDED',
        created_at=1700000000.123456,
        user_id='user-1a2b3c4d',
        return_value=return_value,
        error='null',
        pid=12345,
        schedule_type='long',
        user_name='alice',
        cluster_name=None,
        status_msg='',
        should_retry=False,
        finished_at=1700000001.654321,
        file_mounts_blob_id=None,
    )


def bench(fn, iters: int, repeats: int = 7):
    for _ in range(min(iters, 5000)):  # warmup
        fn()
    per_call_ns = []
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                fn()
            t1 = time.perf_counter_ns()
            per_call_ns.append((t1 - t0) / iters)
    finally:
        gc.enable()
    return min(per_call_ns), statistics.mean(per_call_ns)


def main():
    # Confirm the dispatch each route resolves to (mirrors routing.py).
    print(f'/orjson  use_dump_json='
          f'{isinstance(_ROUTE_ORJSON.response_class, DefaultPlaceholder)}  '
          f'response_class={_ROUTE_ORJSON.response_class}')
    print(f'/default use_dump_json='
          f'{isinstance(_ROUTE_DEFAULT.response_class, DefaultPlaceholder)}  '
          f'response_class={_ROUTE_DEFAULT.response_class}\n')

    cases = [
        ('tiny    (0 clusters)', 0, 200_000),
        ('small   (10 clusters)', 10, 100_000),
        ('medium  (200 clusters)', 200, 20_000),
        ('large   (2000 clusters)', 2000, 3_000),
    ]

    print(f'{"payload":<26}{"bytes":>9} | '
          f'{"orjson ns":>11}{"default ns":>12} | '
          f'{"default/orjson":>15}{"orjson MB/s":>13}{"default MB/s":>13}')
    print('-' * 116)

    for label, n, iters in cases:
        value = validate(_ROUTE_DEFAULT, make_payload(n))

        oj = lambda: serialize_via_route(_ROUTE_ORJSON, value)
        df = lambda: serialize_via_route(_ROUTE_DEFAULT, value)

        assert orjson.loads(oj()) == orjson.loads(df()), f'mismatch for {label}'
        nbytes = len(df())

        oj_min, _ = bench(oj, iters)
        df_min, _ = bench(df, iters)
        ratio = df_min / oj_min
        oj_mbps = nbytes / oj_min * 1e9 / 1e6
        df_mbps = nbytes / df_min * 1e9 / 1e6
        print(f'{label:<26}{nbytes:>9} | '
              f'{oj_min:>11.0f}{df_min:>12.0f} | '
              f'{ratio:>14.2f}x{oj_mbps:>13.0f}{df_mbps:>13.0f}')

    print('\nBOTH arms sourced from real route objects (response_field + '
          'response_class).\ndefault/orjson < 1.0 means the default '
          '(Pydantic serialize_json) path is faster.')


if __name__ == '__main__':
    main()
