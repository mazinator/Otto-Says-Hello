**UNDER CONSTRUCTION**

# Otto-Says-Hello

Just Otto trying to beat everyone in Othello


Otto is an AI bot trained with deep reinforcement learning to destroy any hopes of ever beating him in Othello. 


TODO: check out state of the art performances, 2 papers raussuchen, value functions bewerten


After watching a few "how to beat everyone in Othello"-videos, the following seem to be the most important aspects for human players:

* try to control the inner 4x4 matrix
* try evolving to the wall, as these spots (especially the corners) have a big advantage
* don't waste your move on non-flipping actions
* having less pieces than the opponent is not a bad thing, if parts of the inner 4x4 and a few spots on the walls are presevered. Having more pieces leads to more possible positions from where stones can be flipped.


It can be concluded that the Othello game has the following properties / leads to the following conclusions:
1. It is markov (the future is only dependent on the current state, not the past)
2. Its state space is too big for a tabular solution
3. It is not possible to calculate all possible states
4. It is a game where short-term decisions can heavily influence the final outcome
5. Decisions about contribution to reward probably should not be made before the end of the episode; Besides winning, the only other two options of assigning rewards would be either for flipping a lot of stones or gaining a particular strong position on the field.

Oh noo, such a complex problem. So what should we do? Neural nets to the rescue :rocket:

Otto's goal is to beat the current performances of Othello-bots, with respect to the computational cost. (Bring you own method)

The following papers seem as an interesting starting point into this subject:
- [OLIVAW: Mastering Othello without Human Knowledge, nor a Fortune](https://arxiv.org/abs/2103.17228). This paper from 2021 aims to minimze the computational power needed to achieve the performance of the state of the art, which covers the aspect of keeping computational cost as low as possible.
- [Deep reinforcement Learning Using Monte-Carlo Tree Search for Hex and Othello](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2777474).
- [Hierarchical Reinforcement Learning for the Game of Othello](https://ir.canterbury.ac.nz/bitstreams/51750833-63c1-42a6-b7bf-08ecd4c58434/download)
- [Q-learning adaptations in the game Othello](https://fse.studenttheses.ub.rug.nl/id/eprint/23027)

A few more interesting links on this topic:
- [GymOthelloEnv](https://github.com/lerrytang/GymOthelloEnv)

Data of real played othello games are rather easy to find on the web, e.g. [https://www.kaggle.com/datasets/andrefpoliveira/othello-games](https://www.kaggle.com/datasets/andrefpoliveira/othello-games).

However, three different approaches will be implemented:
- train on human-generated games only
- train on simulated games only
- train on human-generated games, afterwards on simulated ones


## Intended work-packages

- Read through papers above to fully grasp state-of-the-art-approaches (10h)
- Set up infrastructure (15h)
- Implement SOFA (25h)
- Implement the approaches mentioned above (10h each)
- Finetune Neural Net, try out different architectures (25h)
- Using an existing application to showcase the results (2)
- Final report and presentation (5h)
