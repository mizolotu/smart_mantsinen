import gym

from typing import Union
from copy import deepcopy
from common.base_vec_env import VecEnv, VecEnvWrapper
from common.vec_normalize import VecNormalize

def unwrap_vec_normalize(env: Union[gym.Env, VecEnv]) -> Union[VecNormalize, None]:
    """
    :param env: (Union[gym.Env, VecEnv])
    :return: (VecNormalize)
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


# Define here to avoid circular import
def sync_envs_normalization(env: Union[gym.Env, VecEnv], eval_env: Union[gym.Env, VecEnv]) -> None:
    """
    Sync eval and train environments when using VecNormalize

    :param env: (Union[gym.Env, VecEnv]))
    :param eval_env: (Union[gym.Env, VecEnv]))
    """
    env_tmp, eval_env_tmp = env, eval_env
    # Special case for the _UnvecWrapper
    # Avoid circular import
    from common.base_class import _UnvecWrapper
    if isinstance(env_tmp, _UnvecWrapper):
        return
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            # sync reward and observation scaling
            eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
        env_tmp = env_tmp.venv
        # Make pytype happy, in theory env and eval_env have the same type
        assert isinstance(eval_env_tmp, VecEnvWrapper), "the second env differs from the first env"
        eval_env_tmp = eval_env_tmp.venv