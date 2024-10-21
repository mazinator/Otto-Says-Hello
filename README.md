**UNDER CONSTRUCTION**

# Otto-Says-Hello

Just Otto trying to beat everyone in Othello


Otto is an AI bot trained with deep reinforcement learning to destroy any hopes of ever beating him in Othello. 


TODO: check out state of the art performances, 2 papers raussuchen, value functions bewerten


After watching a few "how to beat everyone in Othello"-videos, the following seem to be the most important aspects:


* try to control the inner 4x4
* try evolving to the wall, as these spots (especially the corners) have a big advantage
* having less pieces than the opponent is really not a bad thing, if parts of the inner 4x4 and a few spots on the walls are presevered. Having more pieces leads to more possible position from where stones can be flipped.


It can be concluded that the Othello game has the following properties / leads to the following conclusions:
1. It is markov (the future is only dependent on the current state, not the past)
2. Its state space is too big for a tabular solution
3. It is not possible to calculate all states
4. It is a game where short-term decisions can heavily influence the final outcome
5. Decisions about contribution to reward probably should not be made before the end of the episode; Besides winning, the only other two options of assigning rewards would be either for flipping a lot of stones or gaining a particular strong position on the field. Therefore, TD-learning is not applicable.

So what should we do? Neural nets to the rescue! :rocket:

The following two papers seem as an interesting starting point into this subject:
[OLIVAW: Mastering Othello without Human Knowledge, nor a Fortune](https://arxiv.org/abs/2103.17228)

