# Otto-Says-Hello

**Sarcasm and overconfidence: ON**

Otto is an AI bot trained with deep reinforcement learning to destroy any hopes of ever beating him in Othello.
**Sarcasm: OFF**

After watching a few "how to beat everyone in Othello"-videos, the following seem to be the most important aspects for human players:

* try to control the inner 4x4 matrix
* try evolving to the wall, as these spots (especially the corners) have a big advantage
* having less pieces than the opponent is not a bad thing, if parts of the inner 4x4 and a few spots on the walls are presevered. Having more pieces 
  leads to more possible positions from where stones can be flipped.


It can be concluded that the Othello game has the following properties / leads to the following conclusions:
1. It is markov (the future is only dependent on the current state, not the past)
2. Its state space is too big for a tabular solution, i.e. it is computationally not feasible to calculate all states (to be more precise, at least for a 10x10 field)
4. It is a game where short-term decisions can heavily influence the final outcome (e.g., not placing a stone in a corner if possible highly significantly 
   reduces the chance of winning)
5. Decisions about contribution to reward probably should not be made before the end of the episode; Besides winning, the only other two options I can 
   think of for assigning rewards would be either for flipping a lot of stones or gaining a particular strong position on the field.

Note: 8x8 Othello is claimed to be solved by a preprint from 2023. Otto will to be trained starting on a 8x8 configuration, other configurations will be 
tryed out depending on the computational resources.

Otto's goal is to beat the current performances of Othello-bots, with respect to the computational cost. (Bring you own method)

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
- Implement the approaches mentioned above (originally intended: 10h each. Took me around 5h to implement a replay buffer from existing games, however 
  never really used because alpha zero, the only architecture I have implemented is not based on a replay buffer but on Monte Carlo Tree Search)
- Finetune Neural Net, try out different architectures (intended: 35h. Did not try out different architectures but definitely more than 35h)
- Debugging (intended: More on the section below on 'How this project turned out')
- Using an existing application to showcase the results (intended: 5h. TBD for final presentation)
- Final report and presentation (intended: 5h. TBD for final presentation)

## Deliverables for Submission 2

### File descriptions

While there should be enough documentation inside the files, here is a high level description of the various files:

#### agents

- simple_agents.py: some dummy baseline agents for testing basic behaviour of the environment. Also led to some interesting insights about the game, 
  more on that later.
- medium_agents.py: not really used, this was during a time period where I didn't realize yet that the state-action table for the simple Q-learning 
  agent was even more stupid than most dummy baselines.
- alpha_zero.py: contains the architecture from the original [AlphaZero](https://arxiv.org/pdf/1712.01815) architecture

#### environment

- board.py: OthelloBoard with all necessary functionalities necessary for playing and training
- replay_buffer.py: A replay buffer based on the kaggle-data from above. Not used however, as I only implemented the alpha zero architecture, 
  which is not based on past experience but learns the game from scratch without any input besides a state representation and the available 
  actions in the current state

#### test

- test.py: This file contains all tests created by me.

#### train

- train_alphazero.py: desired configuration of the architecture, optimizer, batch size and so on. Automatically tries to retrieve past checkpoints 
  created with this configuration; either starts training from scratch or from last checkpoint.

#### utils

- agent\_vs\_agent\_simulation.py: contains a function which simulates n games between 2 agents with a defined first player (always black/white or alternating). 
  Stores the results afterwards in /out/agent\_vs\_agent\_results.csv.
- data_loader.py: loads and returns the data used in the file replay_buffer.py, also was able to load the simple Q-learning agent.
- mcts.py: Contains the monte carlo tree search used in the AlphaZero model to estimate the value of the currently available state-action pairs
- model_loader.py: Loads the latest alpha-zero model from checkpoint if available
- nn_helpers.py: two functions which incldue a get_device() function and a transform-function from board to tensor (a board is represented by a 8x8 grid with 
  each field being either black, white or empty. The input for AlphaZero are 3 8x8 tensors + the current player)
- results_writer.py: writes stuff to agent_vs_agent_results.csv


### Description of end-to-end pipeline

In general, I tried to generalize my code as much as possible, e.g. train_alphazero.py is basically a generic training class besides the loss function which I 
had to implemented on my own. One could just put an if-bracket above it and use
a parameter for it, but I didn't get this far. The main-function in this file can be used to load different models, optimizers and checkpoints if available.





### Error metric

I have done a lot of research on error metrics for reinforcement learning, and my final conclusion is that it makes no sense to use a different 
metric for the AlphaZero-architecture. For neural networks, there is no one-size-fits-all solution; my feeling is that this is even more the
case for reinforcement learning. Also, I did not find a lot of research on the various impacts of different error metrics for different architectures
in this area. Trying out multiple different error metrics was just not possible given the time I needed to train a single model on my laptop.
I also assume that authors of such papers which did have such resources used the error metric which worked the best. The arguments above were enough
for me to proceed with the original error metric provided by the authors.

The loss function is defined as: $L=(z-v)^2-\pi^T*log(p)+\lVert\theta\rVert^2$

It can be split up into the following parts:

$(z-v)^2$: this term measures the error between the predicted game outcome (v) and the actual game outcome (z). It is basically an MSE which is pushing 
the neural network towards accurately predicting game outcomes.

$-\pi^T*log(p)$: This term measures the divergence between the predicted move probabilities (p) and the improved policy $\pi$ derived from Monte Carlo 
Tree Search. This part based on negative log-likelihood pushes the networks predicted move towards the improved policy.

$\lVert\theta\rVert^2$: the L2-regularization term penalizes large weights in the neural network, which as far as I understand is especially important 
in such a small network.


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
  training was shit for like 2 million episode. Always print the loss... (8.12.2024)
- I will try to fix the issue until tomorrow and then start training for the last time, afterwards I just accept whatever the result is. (8.12.2024) 

# How this project turned out

The magnitude of how much I underestimated the work necessary to reach my original goals is laughable. They maybe would have been somewhat realistic if it was my 5th 
big reinforcement project. Excluding sandboxes, this was my first bigger reinforcement project. And I wouldn't necessarily say that the neural network itself was 
the main challenge (even though I have once again learned various new aspects of them), but rather the significantly different approach of implementation
compared to (un)supervised learning. Before even starting with the alpha zero architecture mid-november, the level of research and learning I had just on the 
topic of reinforcement learning was really unexpected given that I already had a solid theoretical background based on the [Sutton&Barto](http://incompleteideas.net/book/the-book-2nd.html) 
book. For example, I spent 2-3 hours in understanding the MuZero architecture before realising that it is an overkill for problems without hidden 
states such as Blackjack. I have spent about 10 hours just into researching different theoretical approaches just to realize that I don't have the computational 
resources for most of them. I had a lot of long nights and weekends on this topic, read papers, watched videos (shoutout to [Yannic Kilcher](https://www.youtube.com/@YannicKilcher)),
and in the end I'm just happy that I learned a lot and have at least one solid agent based on the original AlphaZero architecture.
\\\\
Overall, I learned a lot about constructing good test cases, I was able to strengthen my experience in reinforcement learning, I had a lot of fun playing 
and experimenting in my environment, and I played a ton of Othello games which was also fun. 
\\\\
Feel free to shit-talk my repo and workflows, I appreciate the time taken and discovering a few aspects which I don't know yet :-)
\\\\
As always when one tries to deep-dive into a topic: now I know that I know nothing.