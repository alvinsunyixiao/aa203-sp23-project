import typing as T

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from flax import struct

##################################################################
# Dynamical Model
#   states = [x, y, theta]
#   controls = [v, w]
##################################################################

@struct.dataclass
class CtrlAffineSys:
    state_lim: jax.typing.ArrayLike
    control_lim: jax.typing.ArrayLike
    f: T.Callable[[jax.typing.ArrayLike], jax.Array] = struct.field(pytree_node=False)
    g: T.Callable[[jax.typing.ArrayLike], jax.Array] = struct.field(pytree_node=False)
    d: T.Callable[[jax.typing.ArrayLike], jax.Array] = struct.field(pytree_node=False)
    sample: T.Callable[[jax.random.PRNGKey, int], jax.Array] = struct.field(pytree_node=False)

    @property
    def state_dim(self) -> jax.Array:
        return len(self.state_lim)

    @property
    def control_dim(self) -> jax.Array:
        return len(self.control_lim)

    def dynamics(self, states: jax.typing.ArrayLike, controls: jax.typing.ArrayLike):
        return self.f(states) + (self.g(states) @ controls[..., jnp.newaxis])[..., 0]

    def is_safe(self, states: jax.typing.ArrayLike, safe_radius: float = 0.7) -> jax.Array:
        return jnp.all(states <= self.state_lim, axis=-1) & \
               jnp.all(states >= -self.state_lim, axis=-1) & \
               (self.d(states) >= safe_radius)

    def is_unsafe(self, states: jax.typing.ArrayLike, unsafe_radius: float = 0.3) -> jax.Array:
        return jnp.any(states > self.state_lim, axis=-1) | \
               jnp.any(states < -self.state_lim, axis=-1) | \
               (self.d(states) < unsafe_radius)

def DoubleIntegrator(state_lim: jax.typing.ArrayLike = np.array([np.inf, np.inf, 1., 1.]),
                     control_lim: jax.typing.ArrayLike = np.array([1., 1.])) -> CtrlAffineSys:
    # x = [x, y, vx, vy]
    # u = [ax, ay]

    def f(states):
        return jnp.zeros_like(states).at[..., :2].set(states[..., 2:])

    def g(states):
        x_zeros = jnp.zeros_like(states[..., 0])
        x_ones = jnp.ones_like(states[..., 0])

        col1 = jnp.stack([x_zeros, x_zeros, x_ones, x_zeros], axis=-1)
        col2 = jnp.stack([x_zeros, x_zeros, x_zeros, x_ones], axis=-1)

        return jnp.stack([col1, col2], axis=-1)

    def d(states):
        return jnp.linalg.norm(states[..., 0:2], axis=-1)

    def sample(rng, num_samples):
        sample_lim = jnp.concatenate([jnp.array([1., 1.]), state_lim[2:] * jnp.sqrt(2)])
        return jax.random.uniform(rng, (num_samples, len(state_lim)),
                                  minval=-sample_lim, maxval=sample_lim)

    return CtrlAffineSys(state_lim=state_lim, control_lim=control_lim, f=f, g=g, d=d, sample=sample)

