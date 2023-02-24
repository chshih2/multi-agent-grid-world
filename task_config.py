from single_agent_task import PushTask, ThrowTask, CatchTask
from multi_agent_task import MultiAgentTask


def get_task(do_training, env_config):
    env_name = env_config['env_name']
    if env_name == "push":
        return PushTask(do_training, env_config)
    elif env_name == "throw":
        return ThrowTask(do_training, env_config)
    elif env_name == "catch":
        return CatchTask(do_training, env_config)
    elif env_name == "multi-agent":
        return MultiAgentTask(do_training, env_config)
    else:
        raise NotImplementedError
