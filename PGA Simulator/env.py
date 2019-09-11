
import gym
import numpy as np
from gym import error, spaces, utils  # are we using spaces?
from gym.utils import seeding
import scipy as sp
from scipy.stats import poisson

class Find_Greatest_Env:

  def __init__(self):
      self.min_bid = 0
      self.max_bid = 50000

      self.min_player = 0
      self.max_player = 1

      self.min_time_lb = 0
      self.max_time_lb = 30

      self.min_cur_time = 0
      self.max_cur_time = 30

      self.low = np.array([self.min_bid, self.min_player, self.min_time_lb, self.min_cur_time])
      self.high = np.array([self.max_bid, self.max_player, self.max_time_lb, self.max_cur_time])


      self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

      self.action_space = [0, 1, 2, 30, 100, 200, 1000, 3000 ]
      #self.action_space = [0, 1]

  def step(self, action, curr_time, state):
      assert 0 <= action <len(self.action_space) , "%r (%s) invalid" % (action, type(action))

      self.state =  state
      #last_bid, player, time_of_last_bid, time_last_state''' = self.state
      last_bid, player, time_of_last_bid, time_last_state = self.state
      done = bool(curr_time == 15)
      self.time_dif = 0

      reward = 0
      if (player == -1): last_bid = 1
      if player == 0 and action == 0:
          self.state = (last_bid, player, time_of_last_bid, curr_time)
          return np.array(self.state), reward, done, {}
      if player != -1 and self.action_space[action] != 0:
          last_bid += self.action_space[action]
          time_of_last_bid = curr_time

      if (action != 0 or done) and player == 1:
          self.time_dif = curr_time - time_of_last_bid
          if  self.time_dif == 1: reward = 1
          elif self.time_dif == 2: reward = 5
          elif self.time_dif == 3: reward = 25
          elif self.time_dif == 4: reward = 40
          elif self.time_dif == 5: reward = 80
          elif self.time_dif == 6: reward = 170
          elif self.time_dif == 7: reward = 350
          elif self.time_dif == 8: reward = 800
          elif self.time_dif == 9: reward = 1700
          elif self.time_dif == 10: reward = 4000
          elif self.time_dif > 11: reward = 9000


      self.state = (last_bid, 1, time_of_last_bid, curr_time)
      return np.array(self.state), reward, done, {}

  def reset(self):

      self.state = np.array([0, -1, 0, 0])
      return np.array(self.state)



