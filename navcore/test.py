from navcore.builder.environment_builder import EnvironmentBuilder

env = EnvironmentBuilder().build_environment()

print(env.robot)
print(env.crowd)
print(len(env.crowd))
print(env.obstacles)
print(len(env.obstacles))
