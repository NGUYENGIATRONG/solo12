from gym_sloped_terrain.envs import solo12_pybulet_env
import numpy as np
action_temp = [0] * 20
robot = solo12_pybulet_env.Solo12PybulletEnv(on_rack=False)
steps = 20000000
action = [0] * 20
amplitude = 0.5
speed = 7
print(robot.BuildMotorIdList())
for step_counter in range(steps):
    time_step = 0.01
    t = step_counter * time_step
    action[1] = action[7] = (np.sin(speed * t) * amplitude + np.pi / 5)
    action[4] = action[10] = -action[1]
    robot.step(action)

