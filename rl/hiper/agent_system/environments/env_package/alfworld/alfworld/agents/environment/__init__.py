def get_environment(env_type):
    if env_type == "AlfredTWEnv":
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        return AlfredTWEnv
    raise NotImplementedError(f"Environment {env_type} is not implemented.")
