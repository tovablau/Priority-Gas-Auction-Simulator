import gym
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math
from env import Find_Greatest_Env

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.00001
GAMMA = 0.99
BATCH_SIZE = 50

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                     state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._control_time_store = []
        self._amt_bid_store = []
        self._auction_store = {}

    def run(self):
        counter = 0
        tot_reward = 0
        time_di = 0
        amt_bid = 0
        while counter < 20:
            state = self._env.reset()
            time = 0
            delta = 4
            k = 0
            self.single_auction_reward = 0
            auction = list()
            p1_moves = list()
            p0_moves = list()
            while True:
                # choose action based on the state
                action = self._choose_action(np.asarray(state))
                # update the state based on the action
                next_state, reward, done, info = self._env.step(action, time, state)

                # if the time is a multiple of blind raising interval give the opportunity for p0 to bid
                if time % delta == 0:
                    next_state, reward, env.time_dif = self._blind_action(state, next_state, reward, env.time_dif, k)
                    k += 1

                print("Amt: {}, Player: {}, Time of lb: {}, Current Time: {}".format(next_state[0], next_state[1], next_state[2], next_state[3]))

                time_di += env.time_dif

                # if p0 made a bid save it
                if next_state[1] == 0 and next_state[2] == next_state[3]:
                    move0 = [next_state[3], next_state[0]]
                    p0_moves.append(move0)
                # if p1 made a move save it
                if next_state[1] == 1 and next_state[2] == next_state[3]:
                    move1 = [next_state[3], next_state[0]]
                    p1_moves.append(move1)

                # save the reward from a single auction
                self.single_auction_reward += reward

                # is the game complete? If so, set the next state to
                # None for storage sake
                if done:
                    amt_bid += next_state[0]
                    next_state = None

                self._memory.add_sample((state, action, reward, next_state))
                self._replay()

                # exponentially decay the eps value
                self._steps += 1
                self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                            * math.exp(-LAMBDA * self._steps)

                # move the agent to the next state and accumulate the reward
                state = next_state
                tot_reward += reward
                # if the game is done, break the loop

                if done:
                    break
                time += 1
            auction.append(p0_moves)
            auction.append(p1_moves)
            self._auction_store[self.single_auction_reward] = auction  # Add new entry

            # print(self.single_auction_reward)
            # print(self.auction_store[self.single_auction_reward])

            print("Step {}, Total reward: {}, Total time dif: {}, Amt bid: {}, Eps: {}".format(self._steps, self.single_auction_reward, time_di, amt_bid, self._eps))
            counter += 1
        # print(gr.auction_store.get(max(gr.auction_store, key=gr.auction_store.get)))

        self._reward_store.append(tot_reward)
        self._control_time_store.append(time_di)
        self._amt_bid_store.append(amt_bid)
    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _blind_action(self, state, next_state, reward, td, k):
        if next_state[3] == 0:
            bid_amt = 1
        else:
            bid_amt = 1 * ((3 + .2) ** k)

        if bid_amt > next_state[0]:
            next_state = (bid_amt, 0, next_state[3], next_state[3])
        elif bid_amt == next_state[0] and next_state[2] == next_state[3]:
            p = random.randint(0, 1)
            next_state = (bid_amt, p, next_state[3], next_state[3])

        if (next_state[0] == bid_amt) and state[1] == 1:
            time_difference = next_state[3] - state[2]
            td = time_difference
            if time_difference == 1:
                reward = 1
            elif time_difference == 2:
                reward = 5
            elif time_difference == 3:
                reward = 25
            elif time_difference == 4:
                reward = 40
            elif time_difference == 5:
                reward = 80
            elif time_difference == 6:
                reward = 170
            elif time_difference == 7:
                reward = 350
            elif time_difference == 8:
                reward = 800
            elif time_difference == 9:
                reward = 1700
            elif time_difference == 10:
                reward = 4000
            elif time_difference > 11:
                reward = 9000
        else: reward = 0

        return next_state, reward, td


    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._control_time_store
    @property
    def amt_bid_store(self):
        return self._amt_bid_store
    @property
    def auction_store(self):
        return self._auction_store

if __name__ == "__main__":
    env = Find_Greatest_Env()

    num_states = env.observation_space.shape[0]
    num_actions = len(env.action_space)

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)

    with tf.Session() as sess:
        sess.run(model.var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON,
                        LAMBDA)
        num_episodes = 4000
        cnt = 0
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
            gr.run()
            cnt += 1

        #print(gr.auction_store)
        plt.plot(gr.reward_store)
        plt.title('Reward')
        plt.show()
        plt.close("all")

        plt.plot(gr.max_x_store)
        plt.title('Control Time')
        plt.show()
        plt.close("all")

        plt.plot(gr.amt_bid_store)
        plt.title('Amount bid')
        plt.show()
        plt.close("all")
        #
        # for key, value in gr.auction_store.items():
        #     print(key, value)
        print(max(k for k, v in gr.auction_store.items()))

        max_r = max(k for k, v in gr.auction_store.items())
        #print(gr.single_auction_reward)
        #print(gr.auction_store[gr.single_auction_reward])
        print(gr.auction_store.get(max_r))
        #plt.plot(gr.auction_store.get(gr.single_auction_reward))
        # g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
        # g2 = (0.4 + 0.3 * np.random.rand(N), 0.5 * np.random.rand(N))
        #
        # data = (g1, g2)
        # colors = ("red", "green")
        # groups = ("player 0", "player 1")
        #
        # # Create plot
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
        #
        # for data, color, group in zip(data, colors, groups):
        #     x, y = data
        # ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
        #
        # plt.title('Matplot scatter plot')
        # plt.legend(loc=2)
        # plt.show()

        t0 = list()
        b0 = list()

        t1 = list()
        b1 = list()


        for i in gr.auction_store.get(max_r)[0]:
            t0.append(i[0])
            b0.append(i[1])

        for i in gr.auction_store.get(max_r)[1]:
            t1.append(i[0])
            b1.append(i[1])

        colors0 = (0, 0, 0)
        colors1 = (1, 0, 0)
        area = np.pi * 3

        # Plot
        plt.scatter(t0, b0, s=area, c=colors0, alpha=0.5)
        plt.scatter(t1, b1, s=area, c=colors1, alpha=0.5)

        # plt.plot(gr.auction_store.get(max(gr.auction_store, key=gr.auction_store.get))[0])
        # plt.plot(gr.auction_store.get(max(gr.auction_store, key=gr.auction_store.get))[1])
        #
        plt.title('Auction with max reward({})'.format(max_r))
        plt.xlabel('Time')
        plt.ylabel('Amount bid')
        #
        plt.show()