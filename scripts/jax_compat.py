"""Compatibility shim: run brax (<= 0.14.2) on JAX >= 0.10.

JAX 0.10 removed `jax.device_put_replicated` and `jax.device_put_sharded`,
but brax's PPO / SAC / APG trainers still call them. Importing this module
restores both, using the drop-in replacements from
https://docs.jax.dev/en/latest/migrate_pmap.html.

If JAX still provides the functions, nothing is changed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _stacked_sharding(devices):
    mesh = Mesh(np.array(devices), ("replica",))
    return NamedSharding(mesh, PartitionSpec("replica"))


def device_put_replicated(x, devices):
    """Copy `x` onto every device, stacked along a new leading axis."""
    devices = list(devices)
    stacked = jax.tree.map(lambda y: jnp.stack([jnp.asarray(y)] * len(devices)), x)
    return jax.device_put(stacked, _stacked_sharding(devices))


def device_put_sharded(shards, devices):
    """Put shards[i] on devices[i], stacked along a new leading axis."""
    devices = list(devices)
    stacked = jax.tree.map(lambda *ys: jnp.stack(ys), *shards)
    return jax.device_put(stacked, _stacked_sharding(devices))


for _name, _fn in (
    ("device_put_replicated", device_put_replicated),
    ("device_put_sharded", device_put_sharded),
):
    try:
        getattr(jax, _name)
    except AttributeError:
        setattr(jax, _name, _fn)
