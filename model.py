import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import typing as T

from jaxopt import OSQP, BoxOSQP
from jax import lax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from dynamics import CtrlAffineSys

class SirenDense(nn.Module):
    feature: int
    is_first: bool = False
    omega0: float = 30.
    param_dtype: T.Any = jnp.float32
    no_activation: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        n = jnp.shape(x)[-1]

        def kernel_init(rng: jax.random.PRNGKey) -> jnp.ndarray:
            if self.is_first:
                scale = jnp.sqrt(6. / n)
            else:
                scale = 1. / n

            return jax.random.uniform(key=rng, shape=(n, self.feature), dtype=self.param_dtype,
                                      minval=-scale, maxval=scale)

        kernel = self.param("kernel", kernel_init)
        if self.is_first:
            kernel *= self.omega0
        bias = self.param("bias", nn.initializers.zeros_init(), (self.feature,), self.param_dtype)

        y = lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())))
        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        if self.no_activation:
            return y
        else:
            return jnp.sin(y)


class SirenMLP(nn.Module):
    features: T.Sequence[int]
    omega0: float = 30.
    no_first_init: bool = False
    param_dtype: T.Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, feat in enumerate(self.features):
            x = SirenDense(feature=feat,
                           is_first=(i == 0) and not self.no_first_init,
                           omega0=self.omega0,
                           param_dtype=self.param_dtype,
                           name=f"fc_{i}")(x)

        x = SirenDense(feature=1,
                       param_dtype=self.param_dtype,
                       no_activation=True,
                       name="fc_out")(x)[..., 0]

        return x


class MLPCertificate(nn.Module):
    features: T.Tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat)(x)
            x = nn.elu(x)

        x = nn.Dense(features=1)(x)[..., 0]

        return x


class NeuralCBF:
    def __init__(self,
        dynamics: CtrlAffineSys,
        cbf_lambda: float = 1.,
        dt: float = 1e-2,
        mlp_configs: T.Tuple[int, ...] = (256, 256, 256, 256),
    ) -> None:
        self.dynamics = dynamics
        self.cbf_lambda = cbf_lambda
        self.dt = dt

        # initialize mlp
        self.mlp_cert = MLPCertificate(features=mlp_configs)
        rng = jax.random.PRNGKey(42)
        params = self.mlp_cert.init(rng, jnp.ones((1, self.dynamics.state_dim)))["params"]
        #tx = optax.chain(
        #    optax.add_decayed_weights(1e-5),
        #    optax.sgd(optax.exponential_decay(1e-2, 4000, 1e-1)),
        #)
        tx = optax.adamw(optax.exponential_decay(1e-3, 4000, 1e-1), weight_decay=1e-5)
        self.state = TrainState.create(apply_fn=self.mlp_cert.apply, params=params, tx=tx)

        # checkpoint manager
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def save(self, ckpt_dir: str, step: int = 0, keep: int = 10):
        save_checkpoint(ckpt_dir=ckpt_dir,
                        target=self.state,
                        step=step,
                        keep=keep,
                        overwrite=True,
                        orbax_checkpointer=self.checkpointer)

    def load(self, ckpt_dir: str, step: T.Optional[int] = None):
        self.state = restore_checkpoint(ckpt_dir=ckpt_dir,
                                        target=self.state,
                                        step=step,
                                        orbax_checkpointer=self.checkpointer)

    def h(self, states):
        return self.state.apply_fn({"params": self.state.params}, states)

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(self, state, u_ref=None, relaxation_penalty=1e4, u_init=None):
        sol, (h, hdot) = self.solve_CBF_QP(self.h, state, u_ref=u_ref, relaxation_penalty=relaxation_penalty, u_init=u_init, maxiter=10000)
        return sol.params.primal[:-1], (h, hdot, sol)

    def solve_CBF_QP(self, h_func, state, relaxation_penalty=1e4, u_ref=None, u_init=None, maxiter=1000):
        h, h_D_x_s = jax.value_and_grad(h_func)(state)

        Lf_h = h_D_x_s @ self.dynamics.f(state)
        Lg_h_c = h_D_x_s @ self.dynamics.g(state)

        ur_dim = self.dynamics.control_dim + 1

        # objective
        def matvec_Q(_, x):
            Q_vec = jnp.ones_like(x).at[-1].set(relaxation_penalty)
            return 2. * Q_vec * x

        if u_ref is None:
            u_ref = jnp.zeros(self.dynamics.control_dim)
        c = jnp.concatenate([-2. * u_ref, jnp.array([0.])])

        # inequality
        def matvec_G(Lg_h, x):
            # CBF QP
            y1 = Lg_h.T @ x[:-1] - x[-1]
            # control constraints
            y_low = -x[:-1]
            y_high = x[:-1]

            return jnp.concatenate([y1[jnp.newaxis], y_low, y_high])

        h1 = -self.cbf_lambda * h - Lf_h
        ulim_low = -self.dynamics.control_lim
        ulim_high = self.dynamics.control_lim
        h_qp = jnp.concatenate([h1[jnp.newaxis], -ulim_low, ulim_high])

        # solve OSQP
        if u_init is None:
            u_init = u_ref
        qp = OSQP(matvec_Q=matvec_Q, matvec_G=matvec_G, maxiter=maxiter)
        init_params = qp.init_params(jnp.concatenate([u_init, jnp.array([0.])]),
                                     (None, c), None, (Lg_h_c, h_qp))
        sol = qp.run(init_params, params_obj=(None, c), params_ineq=(Lg_h_c, h_qp))

        u = sol.params.primal[:-1]
        hdot = Lf_h + Lg_h_c @ u

        return sol, (h, hdot)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _cbf_train_step(self,
                        states: jax.Array,
                        controls: jax.Array,
                        state: TrainState,
                        step: int) -> TrainState:
        def loss_fn(params):
            h_func = lambda x: state.apply_fn({"params": params}, x)

            def solve_single(x):
                sol, (h, _) = self.solve_CBF_QP(h_func, x)
                success = (sol.state.status == BoxOSQP.SOLVED)

                return h, sol.params.primal[-1], success

            h_b, r_b, success_b = jax.vmap(solve_single)(states)
            is_safe = self.dynamics.is_safe(states)
            is_unsafe = self.dynamics.is_unsafe(states)

            eps = 1e-2
            loss_safe_b = jnp.where(is_safe, jnp.maximum(h_b + eps, 0.), 0.)
            loss_unsafe_b = jnp.where(is_unsafe, jnp.maximum(-h_b + eps, 0.), 0.)

            loss_safe = jnp.sum(loss_safe_b) / jnp.sum(is_safe)
            loss_unsafe = jnp.sum(loss_unsafe_b) / jnp.sum(is_unsafe)
            loss_qp = jnp.mean(r_b)

            loss = loss_safe + loss_unsafe + 1e1 * loss_qp

            return loss, {
                "loss": loss,
                "loss_safe": loss_safe,
                "loss_unsafe": loss_unsafe,
                "loss_qp": loss_qp,
                "num_safe": jnp.sum(is_safe),
                "num_unsafe": jnp.sum(is_unsafe),
                "num_success": jnp.sum(success_b),
            }

        grads, metadict = jax.grad(loss_fn, has_aux=True)(state.params)

        return state.apply_gradients(grads=grads), metadict, grads

    def train_step(self, states: jax.Array, controls: jax.Array, step: int) -> T.Dict[str, T.Any]:
        self.state, metadict, grads = self._cbf_train_step(states, controls, self.state, step)
        return metadict

