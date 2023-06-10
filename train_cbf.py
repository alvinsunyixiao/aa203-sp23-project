import argparse
import jax

from jax.config import config

from dynamics import DoubleIntegrator
from model import NeuralCBF

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="ckpts",
                        help="output directory to store checkpoints")
    parser.add_argument("--batch-size", type=int, default=2**14,
                        help="batch size for each training iteration")

    return parser.parse_args()

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    #config.update("jax_debug_nans", True)

    args = parse_args()

    dynamics = DoubleIntegrator()
    cbf = NeuralCBF(dynamics, cbf_lambda=.2)

    rng = jax.random.PRNGKey(42)

    for i in range(10000):
        rng, key = jax.random.split(rng)
        states = dynamics.sample(key, args.batch_size)
        rng, key = jax.random.split(rng)
        controls = jax.random.uniform(key, (args.batch_size, dynamics.control_dim), minval=-dynamics.control_lim, maxval=dynamics.control_lim)
        metadict = cbf.train_step(states, controls, i)
        print(f"Step: {i} Total Loss: {metadict['loss']:.4f} Safe Loss: {metadict['loss_safe']:.4f} Unsafe Loss: {metadict['loss_unsafe']:.4f} QP Loss: {metadict['loss_qp']:.4f} Num Sucess: {metadict['num_success']}")

        if i % 100 == 0:
            cbf.save(args.output, step=i)
