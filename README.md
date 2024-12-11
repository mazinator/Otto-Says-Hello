# Otto-Says-Hello

**Sarcasm and overconfidence: ON**

Otto is an AI bot trained with deep reinforcement learning to destroy any hopes of ever beating him in Othello.
**Sarcasm: OFF**

After watching a few "how to beat everyone in Othello"-videos, the following seem to be the most important aspects for human players:

* try to control the inner 4x4 matrix
* try evolving to the wall, as these spots (especially the corners) have a big advantage
* having less pieces than the opponent is not a bad thing, if parts of the inner 4x4 and a few spots on the walls are presevered. Having more pieces 
  leads to more possible positions from where stones can be flipped.


One can derive that the Othello game has the following properties:
1. It is markov (the future is only dependent on the current state, not the past)
2. Its state space is too big for a tabular solution, i.e. it is computationally not feasible to calculate all states (to be more precise, at least for a 10x10 field)
4. It is a game where short-term decisions can heavily influence the final outcome (e.g., not placing a stone in a corner if possible highly significantly 
   reduces the chance of winning)
5. Decisions about contribution to reward probably should not be made before the end of the episode; Besides winning, the only other two options I can 
   think of for assigning rewards would be either for flipping a lot of stones or gaining a particular strong position on the field.

Note: 8x8 Othello is claimed to be solved by a preprint from 2023. Otto will to be trained starting on a 8x8 configuration, other configurations will be 
tryed out depending on the computational resources. (Update from future me: 8x8 enough for my MacBook; main drawback is the MCTS simulation, not the model itself)

Otto's goal is to beat the current performances of Othello-bots, with respect to the computational cost. (Bring you own method)
(Update from future me: far off haha, I was able to successfully implement the original AlphaZero architecture)

The following papers seem as an interesting starting point into this subject:
- [OLIVAW: Mastering Othello without Human Knowledge, nor a Fortune](https://arxiv.org/abs/2103.17228). This paper from 2021 aims to minimze the computational power needed to achieve 
  the performance of the state of the art, which covers the aspect of keeping computational cost as low as possible.
- [Deep reinforcement Learning Using Monte-Carlo Tree Search for Hex and Othello](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2777474).
- [Hierarchical Reinforcement Learning for the Game of Othello](https://ir.canterbury.ac.nz/bitstreams/51750833-63c1-42a6-b7bf-08ecd4c58434/download)
- [Q-learning adaptations in the game Othello](https://fse.studenttheses.ub.rug.nl/id/eprint/23027)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461)

Another interesting link on this topic:
- [GymOthelloEnv](https://github.com/lerrytang/GymOthelloEnv)

Data of real played othello games are rather easy to find on the web, e.g. [https://www.kaggle.com/datasets/andrefpoliveira/othello-games](https://www.kaggle.com/datasets/andrefpoliveira/othello-games).

However, three different approaches will be implemented:
- train on human-generated games only
- train on simulated games only
- train on human-generated games, afterwards on simulated ones


## Work-packages

- Read through papers above to fully grasp state-of-the-art-approaches (originally intended: 10h. actually: idk I lost track of the time as watched also 
  a lot of content on YouTube, so probably way more than 10 hours(
- Set up infrastructure (originally intended: 15h. Actually: creating the board, a lot of tests and a few simple baseline agents took around 15 hours)
- Implement SOFA (Q-learning) (originally intended: 15h. Completely skipped that part, as those architecture were too expensive for my computer)
- Implement the approaches mentioned above (originally intended: 10h each. Decided against separate approaches, took around 10 hours)
- Finetune Neural Net, try out different architectures (intended: 35h. Did not try out different architectures but definitely more than 35h)
- Debugging (not planned originally. More on the section below on 'How this project turned out')
- Implementation of MCTS (not planned originally. Lost a day on a somewhat efficient implementation before realizing that there are MCTS-packages available)
- Using an existing application to showcase the results (intended: 5h. TBD for final presentation)
- Final report and presentation (intended: 5h. TBD for final presentation)

## Deliverables for Submission 2

### File descriptions

While there should be enough documentation inside the files, here is a high level description of the various files:

### play.py

This class can be used to play against an agent of your choice, check out parse_args() function for available arguments.

#### agents

- simple_agents.py: some dummy baseline agents for testing basic behaviour of the environment. Also led to some interesting insights about the game, 
  more on that later.
- medium_agents.py: not really used, this was during a time period where I didn't realize yet that the state-action table for the simple Q-learning 
  agent was even more stupid than most dummy baselines.
- alpha_zero.py: contains the architecture from the original [AlphaZero](https://arxiv.org/pdf/1712.01815) architecture

#### environment

- board.py: OthelloBoard with all necessary functionalities necessary for playing and training
- replay_buffer.py: Contains 2 replay buffers; one would be for the kaggle data, the other one is used for AlphaZero (different requirements than first one)

#### test

- test.py: This file contains all tests created by me to verify a correct implementation of the environment, data loading and dummy agents.

#### train

- train_alphazero.py: desired configuration of the architecture, optimizer, batch size and so on. Automatically tries to retrieve past checkpoints
  (model and replay_buffer) created with this configuration; either starts training from scratch or from last checkpoint.

#### utils

- agent\_vs\_agent\_simulation.py: contains a function which simulates n games between 2 agents with a defined first player (always black/white or alternating). 
  Stores the results afterwards in /out/agent\_vs\_agent\_results.csv.
- data_loader.py: loads and returns the data used in the file replay_buffer.py, also was able to load the simple Q-learning agent.
- mcts_wrapper.py: Contains the monte carlo tree search used in the AlphaZero model to estimate the value of the currently available state-action pairs
- model_loader.py: Loads the latest alpha-zero model from checkpoint if available
- nn_helpers.py: two functions which incldue a get_device() function and a transform-function from board to tensor (a board is represented by a 8x8 grid with 
  each field being either black, white or empty. The input for AlphaZero are 3 8x8 tensors + the current player)
- results_writer.py: writes stuff to agent_vs_agent_results.csv


### Description of end-to-end pipeline

In general, I tried to generalize my code as much as possible, e.g. train_alphazero.py is more or less a generic training class (maybe minor changes 
necessary for different models). The main-function in this file can be used to load different models, optimizers and checkpoints if available, and
all relevant hyperparameters for the model and the MCTS.

Quick list of the parameters to set in train_agent() function:
* board: OthelloBoard (currently, must be an 8x8 board but can be adjusted easily if desired)
* model: Either AlphaZeroNet or AlphaZeroNetWithResiduals from alpha_zero.py
* optimizer: e.g. Adam
* episodes: max episodes (by default max.integer, doesn't stop training)
* batch_size: batch_size (sampled from the ReplayBuffer)
* checkpoint_interval: after this many episodes, current model and replay_buffer and will be saved.
* model_loaded: boolean if training is from scratch or not, relevant for correct projection of current training episode
* episode_loaded: the number of already trained episodes if model_loaded=True
* c: hyperparameter for the regularization term in the loss function, the higher the more penalization for more extreme parameters
* simulations_between_training: number of simulated games before the model is trained from the ReplayBuffer
* epochs_for_batches: the number of epochs the model is trained (every 'simulations_between_training'-times)
* mcts_max_time: hyperparameter for MCTS. the maximum time in seconds for the MCTS simulator package (e.g., with mostly 
  60 actions per game, 1000 would lead to around 1min for one simulated game)
* mcts_exploration_constant: hyperparameter for MCTS. constant used for UCB-estimation in MCTS
* replay_buffer_folder_path: folder path to replay buffer file
* replay_buffer_in: file name to read in replay buffer
* replay_buffer_out: file name to write out replay buffer



### Error metric

I have done a lot of research on error metrics for reinforcement learning, and my final conclusion is that it makes no sense to use a different 
metric for the AlphaZero-architecture. For neural networks, there is no one-size-fits-all solution; my feeling is that this is even more the
case for reinforcement learning. Also, I did not find a lot of research on the various impacts of different error metrics for different architectures
in this area. Trying out multiple different error metrics was just not possible given the time I needed to train a single model on my laptop.
I also assume that authors of such papers which did have such resources used the error metric which worked the best. The arguments above were enough
for me to proceed with the original error metric provided by the authors.

The loss function is defined as: $L=(z-v)^2-\pi^T*log(p)+\lVert\theta\rVert^2$

It can be split up into the following parts:

$(z-v)^2$: value loss. this term measures the error between the predicted game outcome (v) and the actual game outcome (z). It is basically an MSE which is pushing 
the neural network towards accurately predicting game outcomes.

$-\pi^T*log(p)$: policy loss. This term measures the divergence between the predicted move probabilities (p) and the improved policy $\pi$ derived from Monte Carlo 
Tree Search. This part based on negative log-likelihood pushes the networks predicted move towards the improved policy.

$\lVert\theta\rVert^2$: regularization. the L2-regularization term penalizes large weights in the neural network, which as far as I understand is especially important 
in such a small network.

It is a little hard to define an expected loss beforehand for an error metric which is composed of 3 different parts with different natures, this was my
best initial estimate:

1. $(z-v)^2$ should be expected to fall under 0.01 
2. $-\pi^T*log(p)$ can be expected to be at around 0.5
3. $\lVert\theta\rVert^2$ is hard to estimate; I found suggestions of it being around 5% of the combined value and policy loss.

Combining these 3 assumptions lead to an estimated error of around 0.6 or so being 'good'. This also means that the MSE part should 
naturally be very low rather quickly and does not play a role afterwards. The regularization term only plays a significant role
if some very big parameters are derived from the optimizer. So the biggest chunk of the work goes to pushing the network towards
the improved policy, which makes sense at least on an abstract level.

A few weeks later. Actually achieved error metric: 

- value loss fell under 0.00001
- policy loss stayed constant at around 1.7 to 2.3 (but at a stage where the policy tensor was 1-hot-encoded from eOthello games, 
  not a probability distribution from MCTS, so makes sense that there is no better fit)
- regularization loss stayed constant at around 0.1 to 0.2


## A few remarks every now and then regarding my progress.

- I seem to have underestimated the level of computational power needed for training an Othello-bot even with limited resources. (18.11.2024)
- I did loose some time while implementing the simple Q-learning agent, as I thought that the training was broken. However, after a little
  more research, it seems that indeed even 2 full days of training is not enough for this Q-Agent to even consistenly beat the RandomAgent, while
  loosing big time against the MaxAgent. (16.11.2024)
- A very popular strategy in (standard) Othello is to perform moves at the beginning where not a lost of disks are flipped, as this limits the
  number of moves the opponent can make. This probably leads to the player being able to flip a lot of disks at the end of the game. As I already
  spent a lot of time and overnight-laptop-resources on this project, I did not loose time to directly implement such a strategy or even take
  handcrafted features into consideration. (18.11.2024)
- What I did however, was to implement two very simple agents: MinAgent (always choosing the move with the least disks flipped) and the MaxAgent (vice versa).
  At least I personally was very convinced that the MaxAgent should beat the MinAgent all the time, which was also confirmed by my tests. However,
  a fews after, I realized that the MaxAgent always won because the MinAgent started! This can maybe be somehow explained by the fact that always more or
  less the same game is played with those two opponents, however it was still rather unexpected. (18.11.2024)
- Overall, I did have to make some decisions on which parts of the Othello-game to explore, as e.g. handcrafting features alone would have probably been enough to
  fill this project. Naturally, I decided to focus on the 'Deep' part. (18.11.2024)
- Storing checkpoint for AlphaZero is really nice, takes just around 2.5MB per checkpoint. (20.11.2024)
- A quick note on how important the learning rate is: using lr 0.0001 learned basically nothing, while lr 0.001 played a solid round after an hour of training. (21.11.2024)
- My training for alpha-zero was probably broken. After the first 300.000 rounds, the model was actually able to beat me a few times, and it definitely learned the
  importance of corners and walls, as it tried to push me near such places and then taking over my stones directly afterwards. However it just stopped 
  getting smarter from that point forward and even degenerated for a while. I am not even sure if it is really broken or I just need more computational power. I trained
  for around 2.5 million episodes, and it was at the end really smart, however I realized that it can be pushed into two very bad positions (B2, B7, G2, G7) which
  if played smartly almost inevitably leads to a win for me. So is it really broken or just too stupid yet? I have also changed the batch sizes significantly 
  (128 to around 100.000) to see how this influences the outcome, which makes it a little hard to evaluate the separate effects. (26.11.2024)
- Addition to earlier: I have changed the batch size to such a high number because I read a paper where their 'mini-batch' was around 4 Million tuples per iteration.
  As far as I understood batch sizes, the only upper bound set for the batch size is that it shouldn't be the whole data and that it fits into memory. However, this
  is only seems to be a valid assumption for very big networks. For smaller networks such as AlphaZero, one should probably not use such big batch sizes, as it 
  could lead to instable training for this case. (4.12.2024)
- I probably broke something during refactoring the code a few days ago, I just realized that the loss is NaN which is a pretty good explanation on why the 
  training was shit for like 1 million episodes. Always print the loss... (8.12.2024)
- I think I fixed it :) It is really amazing how robust and forgiving neural networks can be - I have no idea why the shit that I called 'code' produced anything
  meaningful at all. Always triple check dimensions and intermediate results. (09.12.2024)
- Looks good. (10.12.2024)
- I fitted around 100 batches of size 32 with rather low quality MCTS-simulations (only around 80 simulations per state) and almost lost against the agent, 
  it definitely something meaningful. (10.12.2024)
- Trained the model on the 25.000 games from eOthello (game from the top 100 players of eOthello) for 100 episodes with batch size 32. It now beats me easily. (11.12.2024)


# How this project turned out

Quick ragequit about myself:
The magnitude of how much I underestimated the work necessary to reach my original goals is laughable. They maybe would have been somewhat realistic if it was my 5th 
big reinforcement project. Excluding sandboxes, this was my first bigger reinforcement project. Before even starting with the alpha zero architecture mid-november, the 
level of research and learning I had just on the topic of reinforcement learning was really unexpected given that I already had a solid theoretical background based 
on the [Sutton&Barto](http://incompleteideas.net/book/the-book-2nd.html) book. For example, I spent 2-3 hours in understanding the MuZero architecture before realising that it is an overkill for problems without 
hidden states such as Blackjack. I have spent about 10 hours just into researching different theoretical approaches just to realize that I don't have the computational 
resources for most of them. I had a lot of long nights and weekends on this topic, read papers, watched videos (shoutout to [Yannic Kilcher](https://www.youtube.com/@YannicKilcher)),
and in the end I'm just happy that I learned a lot and have at least one solid agent based on the original AlphaZero architecture.

Overall, I still feel like it was a great project:


* I learned a lot about constructing good test cases,
* I was able to strengthen my experience in reinforcement learning,
* I learned a lot about neural networks (especially how to effectively handle the dimension and debugging them, but also 
  on tuning hyperparameters and how I can find better hyperparameters),
* I had a lot of fun playing and experimenting in my environment, 
* and I played a ton of Othello games which was also fun.

# Conclusion

Now I know that I know nothing.