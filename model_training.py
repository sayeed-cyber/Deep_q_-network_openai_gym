# trainning the model and saveing it 
from algodqn import DeepQLearning
import gym


env=gym.make('CartPole-v1')
gamma=1
epsilon=0.1
numberEpisodes=10
LearningQDeep=DeepQLearning(env,gamma,epsilon,numberEpisodes)
LearningQDeep.trainingEpisodes()
LearningQDeep.sumRewardsEpisode
LearningQDeep.mainNetwork.summary()
LearningQDeep.mainNetwork.save("Dqnmodel.h5")



