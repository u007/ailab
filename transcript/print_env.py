import os

env_vars = os.environ

# Print each environment variable and its value
for key, value in env_vars.items():
    print(f"{key}: {value}")