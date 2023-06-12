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

    @property
    def state_dim(self) -> jax.Array:
        return len(self.state_lim)

    @property
    def control_dim(self) -> jax.Array:
        return len(self.control_lim)

    def dynamics(self, states: jax.typing.ArrayLike, controls: jax.typing.ArrayLike):
        return self.f(states) + (self.g(states) @ controls[..., jnp.newaxis])[..., 0]

    def is_safe(self, states: jax.typing.ArrayLike, safe_radius: float = 0.6) -> jax.Array:
        return self.d(states) >= safe_radius

    def is_unsafe(self, states: jax.typing.ArrayLike, unsafe_radius: float = 0.2) -> jax.Array:
        return self.d(states) < unsafe_radius

    def sample(self, rng: jax.random.PRNGKey, num_samples: int) -> jax.Array:
        return jax.random.uniform(rng, (num_samples, self.state_dim),
                                  minval=-self.state_lim, maxval=self.state_lim)

def DoubleIntegrator(state_lim: jax.typing.ArrayLike = np.array([1., 1., 5., 5.]),
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

    return CtrlAffineSys(state_lim=state_lim, control_lim=control_lim, f=f, g=g, d=d)

def ExtendedUnicycle(state_lim: jax.typing.ArrayLike = np.array([1., 1., -np.pi, 5., 5.]),
        control_lim: jax.typing.ArrayLike = np.array([1., 1.])) -> CtrlAffineSys:
    # x = [x, y, theta, v, w]
    # u = [a, alpha]

    def f(states):
        xdot = states[..., 3] * jnp.cos(states[..., 2])
        ydot = states[..., 3] * jnp.sin(states[..., 2])
        tdot = states[..., 4]
        vdot = jnp.zeros_like(xdot)
        wdot = jnp.zeros_like(xdot)

        return jnp.stack([xdot, ydot, tdot, vdot, wdot], axis=-1)

    def g(states):
        x_zeros = jnp.zeros_like(states[..., 0])
        x_ones = jnp.ones_like(states[..., 0])

        col1 = jnp.stack([x_zeros, x_zeros, x_zeros, x_ones, x_zeros], axis=-1)
        col2 = jnp.stack([x_zeros, x_zeros, x_zeros, x_zeros, x_ones], axis=-1)

        return jnp.stack([col1, col2], axis=-1)

    def d(states):
        return jnp.linalg.norm(states[..., 0:2], axis=-1)

    return CtrlAffineSys(state_lim=state_lim, control_lim=control_lim, f=f, g=g, d=d)
