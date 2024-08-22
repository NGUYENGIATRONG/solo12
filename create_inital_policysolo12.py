import numpy as np
import simulation.solo_pybulet_env as solo12
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

tuned_actions_solo12 = np.array([[0.8, 0.8, 0.8, 0.8,
                                  0.428571429, 0.428571429, 0.428571429, 0.428571429,
                                  0, 0, 0, 0,
                                  -1, -1, -1, -1,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0],

                                 [0.8, 0.8, 0.8, 0.8,
                                  0.428571429, 0.428571429, 0.428571429, 0.428571429,
                                  0, 0, 0, 0,
                                  -1, -1, -1, -1,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0],

                                 [0.8, 0.8, 0.8, 0.8,
                                  0.428571429, 0.428571429, 0.428571429, 0.428571429,
                                  0, 0, 0, 0,
                                  -1, -1, -1, -1,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0]])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--robotName', help='the robot to be trained for', type=str, default='Solo12-v4')
    parser.add_argument('--policyName', help='file name of the initial policy', type=str, default='new_policyname')
    args = parser.parse_args()

    if args.policyName == 'new_policyname':
        args.policyName += args.robotName
    # Number of steps per episode
    num_of_steps = 400

    # list that tracks the states and actions
    states = []
    actions = []
    do_supervised_learning = True

    if args.robotName == 'Solo12-v4':
        # for HyQ
        idx1 = [2]
        idx2 = [0, 3]
        env = solo12.Solo12PybulletEnv(render=True, wedge=True, stairs=False, on_rack=False)

        experiment_counter = 0

        for i in idx1:
            for j in idx2:
                if i == 0 and j == 3:
                    break
                t_r = 0
                env.randomize_only_inclines(default=True, idx1=i, idx2=j)

                cstate = env.reset()
                roll = 0
                pitch = 0

                for ii in np.arange(0, num_of_steps):
                    cstate, r, _, info = env.step(tuned_actions_solo12[experiment_counter])
                    t_r += r
                    states.append(cstate)
                    actions.append(tuned_actions_solo12[experiment_counter])
                experiment_counter = experiment_counter + 1
                print("Returns of the experiment:", t_r)

    if do_supervised_learning:
        model = LinearRegression(fit_intercept=False)
        states = np.array(states)
        actions = np.array(actions)

        # train
        print("Shape_X_Labels:", states.shape, "Shape_Y_Labels:", actions.shape)
        model.fit(states, actions)
        action_pred = model.predict(states)

        # test
        print('Mean squared error:', mean_squared_error(actions, action_pred))
        res = np.array(model.coef_)
        np.save("./initial_policies/" + args.policyName + ".npy", res)
