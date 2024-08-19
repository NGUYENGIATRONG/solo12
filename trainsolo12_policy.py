import os
import gym
# import time
import multiprocessing as mp
from multiprocessing import Pipe
import argparse
import math
# Utils
from SlopedTerrainLinearPolicy.utils.logger import DataLog
from SlopedTerrainLinearPolicy.utils.make_train_plots import make_train_plots_ars
import random
# Registering new environments
from gym.envs.registration import register

import numpy as np

PI = math.pi


# Setting the Hyper Parameters

class HyperParameters:
    """
    This class is basically a struct that contains all the hyperparameters that you want to tune
    """

    def __init__(self, stairs=False, action_dim=10, normal=True, gait='trot', msg='', nb_steps=10000,
                 episode_length=1000, learning_rate=0.02, nb_directions=16, nb_best_directions=8, noise=0.03, seed=1,
                 env_name='HalfCheetahBulletEnv-v0', curilearn=60, evalstep=3):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.nb_directions = nb_directions
        self.nb_best_directions = nb_best_directions
        assert self.nb_best_directions <= self.nb_directions
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
        self.normal = normal
        self.msg = msg
        self.gait = gait
        self.action_dim = action_dim
        self.stairs = stairs
        self.curilearn = curilearn
        self.evalstep = evalstep
        self.domain_Rand = 1
        self.logdir = ""
        self.anti_clock_ori = True

    def to_text(self, path):
        res_str = ''
        res_str = res_str + 'learning_rate: ' + str(self.learning_rate) + '\n'
        res_str = res_str + 'noise: ' + str(self.noise) + '\n'
        if self.stairs:
            res_str = res_str + 'env_name: ' + str(self.env_name) + 'with stairs \n'
        else:
            res_str = res_str + 'env_name: ' + str(self.env_name) + '\n'
        res_str = res_str + 'episode_length: ' + str(self.episode_length) + '\n'
        res_str = res_str + 'direction ratio: ' + str(self.nb_directions / self.nb_best_directions) + '\n'
        res_str = res_str + 'Normal initialization: ' + str(self.normal) + '\n'
        res_str = res_str + 'Gait: ' + str(self.gait) + '\n'
        res_str = res_str + 'Incline_Orientaion_Anti-Clockwise: ' + str(self.anti_clock_ori) + '\n'
        res_str = res_str + 'domain_Randomization: ' + str(self.domain_Rand) + '\n'
        res_str = res_str + 'Curriculmn introduced at iteration: ' + str(self.curilearn) + '\n'
        res_str = res_str + self.msg + '\n'
        fileobj = open(path, 'w')
        fileobj.write(res_str)
        fileobj.close()


# Multiprocess Exploring the policy on one specific direction and over one episode

_RESET = 1
_CLOSE = 2
_EXPLORE = 3


def ExploreWorker(rank, child_Pipe, envname, arg):
    environment = gym.make(envname)
    # nb_inputs = environment.observation_space.sample().shape[0]
    _ = environment.reset()
    n = 0
    while True:
        n += 1
        try:
            # Only block for short times to have keyboard exceptions be raised.
            if not child_Pipe.poll(0.001):
                continue
            message, payload = child_Pipe.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if message == _RESET:
            _ = env.reset()
            child_Pipe.send(["reset ok"])
            continue
        if message == _EXPLORE:
            data = payload[0]
            hp = payload[1]
            direction = payload[2]
            delta = payload[3]
            state = env.reset()
            done = False
            num_plays = 0.
            sum_rewards = 0
            while num_plays < hp.episode_length:
                action = data.evaluate(state, delta, direction, hp)
                state, reward, done, _ = env.step(action)
                sum_rewards += reward
                num_plays += 1
            childPipe.send([sum_rewards, num_plays])
            continue
        if message == _CLOSE:
            childPipe.send(["close ok"])
            break
    childPipe.close()


# Building the AI

class Policy:

    def __init__(self, input_size, output_size, env_name, normal, arg):
        try:
            print("Training from guided policy,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
            self.theta = np.load(arg.policy)
        except:
            print("Training from random policy")
            if normal:
                print("Training from random policy")
                self.theta = np.random.randn(output_size, input_size)
            else:
                self.theta = np.zeros((output_size, input_size))

        self.env_name = env_name
        print("Starting policy theta=", self.theta)

    def evaluate(self, state_input, delta, direction, hp):
        if direction is None:
            return np.clip(self.theta.dot(state_input), -1.0, 1.0)
        elif direction == "positive":
            return np.clip((self.theta + hp.noise * delta).dot(state_input), -1.0, 1.0)
        else:
            return np.clip((self.theta - hp.noise * delta).dot(state_input), -1.0, 1.0)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, direction in rollouts:
            step += (r_pos - r_neg) * direction
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        # timestr = time.strftime("%Y%m%d-%H%M%S")


# Exploring the policy on one specific direction and over one episode

def explore(environment, data, direction, delta, hp):
    # _ = environment.observation_space.sample().shape[0]
    state = environment.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while num_plays < hp.episode_length:
        action = data.evaluate(state, delta, direction, hp)
        state, reward, done, _ = environment.step(action)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards


def policyevaluation(environment, data, hp):
    reward_evaluation = 0

    if hp.domain_Rand:

        # Evaluation Dataset with domain randomization
        # --------------------------------------------------------------
        incline_deg_range = [3, 4]  # 9, 11
        incline_ori_range = [0, 2, 3]  # 0, 60, 90 degree
        fric = [0, 1]  # surface friction 0.55, 0.6
        mf = [0]  # extra mass at front 0gm
        mb = [3]  # extra mass at back 150gm
        ms = [0, 1]  # motorstrength 0.52, 0.6
        ef = [0]  # pertubation force 0
        # --------------------------------------------------------------
        total_combinations = (len(incline_deg_range) * len(incline_ori_range)
                              * len(fric) * len(ms) * len(ef))
        for j in incline_deg_range:
            for i in incline_ori_range:
                for k in fric:
                    for f in mf:
                        for b in mb:
                            for s in ms:
                                for t in ef:
                                    environment.Set_Randomization(default=True, idx1=j, idx2=i, idx3=k, idx0=f, idx11=b,
                                                                  idxc=s,
                                                                  idxp=t)
                                    reward_evaluation = reward_evaluation + explore(environment, data, None, None, hp)

        reward_evaluation = reward_evaluation / total_combinations

    else:
        # Evaluation Dataset without domain randomization
        # --------------------------------------------------------------
        incline_deg_range = [2, 3, 4]  # 11, 13
        incline_ori_range = [0, 2, 3]  # 0, 30, 45 degree
        # --------------------------------------------------------------
        total_combinations = len(incline_deg_range) * len(incline_ori_range)

        for j in incline_deg_range:
            for i in incline_ori_range:
                environment.randomize_only_inclines(default=True, idx1=j, idx2=i)
                reward_evaluation = reward_evaluation + explore(environment, data, None, None, hp)

        reward_evaluation = reward_evaluation / total_combinations

    return reward_evaluation


def create_unique_dir(base_name):
    counter = 0
    dir_name = base_name  # Khởi đầu với tên ban đầu

    # Kiểm tra sự tồn tại của thư mục
    while os.path.exists(dir_name):
        counter += 1
        dir_name = f"{base_name}_{counter}"  # Cập nhật tên với số đếm và dấu chấm

    # Thư mục với tên duy nhất không tồn tại, tạo nó
    os.mkdir(dir_name)
    print(f"Folder '{dir_name}' created.")

    return dir_name


# Training the AI
def train(environment, data, hp, parent_pipes, info):
    # global process_count
    info.logdir = "experiments/" + info.logdir
    logger = DataLog()
    total_steps = 0
    best_return = -99999999

    working_dir = os.getcwd()

    unique_dir_name = create_unique_dir(info.logdir)

    os.chdir(unique_dir_name)

    if os.path.isdir('iterations') is False:
        os.mkdir('iterations')

    if os.path.isdir('logs') is False:
        os.mkdir('logs')
    hp.to_text('hyperparameters')

    log_dir = os.getcwd()
    os.chdir(working_dir)

    for step in range(hp.nb_steps):
        if hp.domain_Rand:
            environment.Set_Randomization(default=False)
        else:
            environment.randomize_only_inclines()
        # Cirriculum learning
        if step > hp.curilearn:
            avail_deg = [9, 11, 13, 13, 15]
            environment.incline_deg = avail_deg[random.randint(0, 4)]
        else:
            avail_deg = [7, 9]
            environment.incline_deg = avail_deg[random.randint(0, 1)]

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = data.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        if parent_pipes:
            p = 0
            process_count = len(parent_pipes)
            while p < hp.nb_directions:
                temp_p = p
                n_left = hp.nb_directions - p  # Số lượng quy trình cần thiết để hoàn thành tìm kiếm

                for k in range(min([process_count, n_left])):
                    parent_pipe = parent_pipes[k]
                    parent_pipe.send([_EXPLORE, [data, hp, "positive", deltas[temp_p]]])
                    temp_p += 1
                temp_p = p
                for k in range(min([process_count, n_left])):
                    positive_rewards[temp_p], step_count = parent_pipes[k].recv()
                    total_steps = total_steps + step_count
                    temp_p = temp_p + 1
                temp_p = temp_p

                for k in range(min([process_count, n_left])):
                    parent_Pipe = parent_pipes[k]
                    parent_Pipe.send([_EXPLORE, [data, hp, "negative", deltas[temp_p]]])
                    temp_p = temp_p + 1
                temp_p = temp_p

                for k in range(min([process_count, n_left])):
                    negative_rewards[temp_p], step_count = parent_pipes[k].recv()
                    total_steps = total_steps + step_count
                    temp_p = temp_p + 1
                temp_p = temp_p + process_count
                print('total steps till now: ', total_steps, 'Processes done: ', temp_p)

        else:
            # Getting the positive rewards in the positive directions
            for k in range(hp.nb_directions):
                positive_rewards[k] = explore(environment, data, "positive", deltas[k], hp)

            # Getting the negative rewards in the negative/opposite directions
            for k in range(hp.nb_directions):
                negative_rewards[k] = explore(environment, data, "negative", deltas[k], hp)

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {
            k: max(r_pos, r_neg)
            for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))
        }
        order = sorted(scores.keys(), key=lambda x: -scores[x])[:int(hp.nb_best_directions)]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array([x[0] for x in rollouts] + [x[1] for x in rollouts])
        sigma_r = all_rewards.std()  # Standard deviation of only rewards in the best directions is what it should be
        # Updating our policy
        data.update(rollouts, sigma_r, info)

        # Start evaluating after only second stage
        if step >= hp.curilearn:
            # policy evaluation after specified iterations
            if step % hp.evalstep == 0:
                reward_evaluation = policyevaluation(environment, data, hp)
                logger.log_kv('steps', step)
                logger.log_kv('return', reward_evaluation)
                if reward_evaluation > best_return:
                    best_policy = data.theta
                    best_return = reward_evaluation
                    np.save(log_dir + "/iterations/best_policy.npy", best_policy)
                print('Step:', step, 'Reward:', reward_evaluation)
                policy_path = log_dir + "/iterations/" + "policy_" + str(step)
                np.save(policy_path, data.theta)

                logger.save_log(log_dir + "/logs/")
                make_train_plots_ars(log=logger.log, keys=['steps', 'return'], save_loc=log_dir + "/logs/")


# Running the main code


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='Gym environment name', type=str, default='solo12')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1234123)
    parser.add_argument('--render', help='OpenGL Visualizer', type=int, default=0)
    parser.add_argument('--steps', help='Number of steps', type=int, default=10000)
    parser.add_argument('--policy', help='Starting policy file (npy)', type=str, default='zeros12x11')
    parser.add_argument('--logdir', help='Directory root to log policy files (npy)', type=str, default='logdir_name')
    parser.add_argument('--mp', help='Enable multiprocessing', type=int, default=1)
    # these you have to set
    parser.add_argument('--lr', help='learning rate', type=float, default=0.2)
    parser.add_argument('--noise', help='noise hyperparameter', type=float, default=0.03)
    parser.add_argument('--episode_length', help='length of each episode', type=float, default=10)
    parser.add_argument('--normal', help='length of each episode', type=int, default=1)
    parser.add_argument('--gait', help='type of gait you want (Only in Stoch2 normal env', type=str, default='trot')
    parser.add_argument('--msg', help='msg to save in a text file', type=str, default='')
    parser.add_argument('--stairs', help='add stairs to the bezier environment', type=int, default=0)
    parser.add_argument('--action_dim', help='degree of the spline polynomial used in the training', type=int,
                        default=20)
    parser.add_argument('--directions', help='divising factor of total directions to use', type=int, default=2)
    parser.add_argument('--curi_learn',
                        help='after how many iteration steps second stage of curriculum learning should start',
                        type=int, default=20)
    parser.add_argument('--eval_step', help='policy evaluation after how many steps should take place', type=int,
                        default=3)
    parser.add_argument('--domain_Rand', help='add domain randomization', type=int, default=1)
    parser.add_argument('--anti_clock_ori', help='rotate the inclines anti-clockwise', type=bool, default=True)

    args = parser.parse_args()
    walk = [0, PI, PI / 2, 3 * PI / 2]
    canter = [0, PI, 0, PI]
    bound = [0, 0, PI, PI]
    trot = [0, PI, PI, 0]
    custom_phase = [0, PI, PI + 0.1, 0.1]
    phase = 0
    if args.gait == "trot":
        phase = trot
    elif args.gait == "canter":
        phase = canter
    elif args.gait == "bound":
        phase = bound
    elif args.gait == "walk":
        phase = walk
    elif args.gait == "custom_phase1":
        phase = custom_phase
    # Custom environments that you want to use ----------------------------------------------------------------------------------------
    register(id=args.env,
             entry_point='gym_sloped_terrain.envs.HyQ_pybullet_env:HyQEnv',
             kwargs={'gait': args.gait, 'render': False, 'action_dim': args.action_dim})
    # ---------------------------------------------------------------------------------------------------------------------------------

    hp = HyperParameters()
    args.policy = './initial_policies/' + args.policy
    hp.msg = args.msg
    hp.env_name = args.env
    print("\n\n", hp.env_name, "\n\n")
    env = gym.make(hp.env_name)
    hp.seed = args.seed
    hp.nb_steps = args.steps
    hp.learning_rate = args.lr
    hp.noise = args.noise
    hp.episode_length = args.episode_length
    hp.nb_directions = int(env.observation_space.sample().shape[0] * env.action_space.sample().shape[0])
    hp.nb_best_directions = int(hp.nb_directions / args.directions)
    hp.normal = args.normal
    hp.gait = args.gait
    hp.action_dim = args.action_dim
    hp.stairs = args.stairs
    hp.curilearn = args.curi_learn
    hp.evalstep = args.eval_step
    hp.domain_Rand = args.domain_Rand
    hp.anti_clock_ori = args.anti_clock_ori
    print("log dir", args.logdir)
    hp.logdir = args.logdir
    np.random.seed(hp.seed)
    max_processes = 20
    parentPipes = None
    if args.mp:
        num_processes = min([hp.nb_directions, max_processes])
        print('processes: ', num_processes)
        processes = []
        childPipes = []
        parentPipes = []

        for pr in range(num_processes):
            parentPipe, childPipe = Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(num_processes):
            p = mp.Process(target=ExploreWorker, args=(rank, childPipes[rank], hp.env_name, args))
            p.start()
            processes.append(p)

    nb_inputs = env.observation_space.sample().shape[0]
    nb_outputs = env.action_space.sample().shape[0]
    policy = Policy(nb_inputs, nb_outputs, hp.env_name, hp.normal, args)
    print("start training")

    train(env, policy, hp, parentPipes, args)

    if args.mp:
        for parentPipe in parentPipes:
            parentPipe.send([_CLOSE, "pay2"])

        # for p in processes:
        #     p.join()
