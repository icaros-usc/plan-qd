"""Hydra utils. Adapted from
https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/release/2023.1.1/omniisaacgymenvs/utils/hydra_cfg/hydra_utils.py"""

from hydra.core.utils import setup_globals
from hydra.utils import get_class, get_object
from omegaconf import OmegaConf


def define_resolvers():
    # Resolvers used in hydra configs (see
    # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
    if not OmegaConf.has_resolver("eq"):
        OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())

    if not OmegaConf.has_resolver("contains"):
        OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())

    if not OmegaConf.has_resolver("if"):
        OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)

    if not OmegaConf.has_resolver("multiply"):
        OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)

    if not OmegaConf.has_resolver("divide"):
        OmegaConf.register_new_resolver("divide", lambda x, y: x / y)

    if not OmegaConf.has_resolver("floordivide"):
        OmegaConf.register_new_resolver("floordivide", lambda x, y: x // y)

    # Allows us to resolve default arguments which are copied in multiple places in the
    # config
    if not OmegaConf.has_resolver("resolve_default"):
        OmegaConf.register_new_resolver(
            "resolve_default", lambda default, arg: default if arg == "" else arg
        )

    # For converting things like numpy.float32 to actual classes
    if not OmegaConf.has_resolver("get_cls"):
        OmegaConf.register_new_resolver(
            name="get_cls", resolver=lambda cls: get_class(cls)
        )

    # For getting non-class objects without type check
    if not OmegaConf.has_resolver("get_obj"):
        OmegaConf.register_new_resolver(
            name="get_obj", resolver=lambda name: get_object(name)
        )

    # Hydra's base resolvers (these seem to not be loaded when experimental reload is
    # used)
    if not OmegaConf.has_resolver("hydra"):
        setup_globals()
