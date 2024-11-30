‼️ **UNDER CONSTRUCTION** ‼️


# Otto-Says-Hello

Otto trying to beat everyone in Othello.


Otto is an AI bot trained with deep reinforcement learning to destroy any hopes of ever beating him in Othello. 

After watching a few "how to beat everyone in Othello"-videos, the following seem to be the most important aspects for human players:

* try to control the inner 4x4 matrix
* try evolving to the wall, as these spots (especially the corners) have a big advantage
* having less pieces than the opponent is not a bad thing, if parts of the inner 4x4 and a few spots on the walls are presevered. Having more pieces leads to more possible positions from where stones can be flipped.


It can be concluded that the Othello game has the following properties / leads to the following conclusions:
1. It is markov (the future is only dependent on the current state, not the past)
2. Its state space is too big for a tabular solution, i.e. it is computationally not feasible to calculate all states (to be more precise, at least for a 10x10 field)
4. It is a game where short-term decisions can heavily influence the final outcome (e.g., not placing a stone in a corner if possible highly significantly reduces the chance of winning)
5. Decisions about contribution to reward probably should not be made before the end of the episode; Besides winning, the only other two options I can think of for assigning rewards would be either for flipping a lot of stones or gaining a particular strong position on the field.

Note: 8x8 Othello is claimed to be solved by a preprint from 2023. Otto will to be trained starting on a 8x8 configuration, other configurations will be tryed out depending on the computational resources.

Otto's goal is to beat the current performances of Othello-bots, with respect to the computational cost. (Bring you own method)

The following papers seem as an interesting starting point into this subject:
- [OLIVAW: Mastering Othello without Human Knowledge, nor a Fortune](https://arxiv.org/abs/2103.17228). This paper from 2021 aims to minimze the computational power needed to achieve the performance of the state of the art, which covers the aspect of keeping computational cost as low as possible.
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


## Intended work-packages

- Read through papers above to fully grasp state-of-the-art-approaches (originally intended: 10h. actually: idk I lost track of the time as watched also a lot of content on YouTube, so probably way more than 10 hours(
- Set up infrastructure (originally intended: 15h. Actually: creating the board, a lot of tests and a few simple baseline agents took around 15 hours)
- Implement SOFA (Q-learning) (15h)
- Implement the approaches mentioned above (10h each)
- Finetune Neural Net, try out different architectures (35h)
- Using an existing application to showcase the results (5h)
- Final report and presentation (5h)

## Deliverables for Submission 2

- Error Metric TODO gibts da überhaupt ah auswahl
- establish working end to end pipeline as a baseline for further experimentation TODO hab i eig schon abgesehen vom NN teil, muss no beschrieben werden
- 

## Interesting aspects arising during implementation (written for Submission 2)

- I seem to have underestimated the level of computational power needed for training an Othello-bot even with limited resources. (18.11.2024)
- I did loose some time while implementing the simple Q-learning agent, as I thought that the training was broken. However, after a little
  more research, it seems that indeed even 2 full days of training is not enough for this Q-Agent to even consistenly beat the RandomAgent, while
  loosing big time against the MaxAgent. (18.11.2024)
- A very popular strategy in (standard) Othello is to perform moves at the beginning where not a lost of disks are flipped, as this limits the
  number of moves the opponent can make. This probably leads to the player being able to flip a lot of disks at the end of the game. As I already
  spent a lot of time and overnight-laptop-resources on this project, I did not loose time to directly implement such a strategy or even take
  handcrafted features into consideration. (18.11.2024)
- What I did however, was to implement two very simple agents: MinAgent (always choosing the move with the least disks flipped) and the MaxAgent (vice versa).
  At least I personally was very convinced that the MaxAgent should beat the MinAgent all the time, which was also confirmed by my tests. However,
  a fews after, I realized that the MaxAgent always won because the MinAgent started! This can maybe be somehow explained by the fact that always more or
  less the same game is played with those two opponents, however it was still very unexpected. (18.11.2024)
- Overall, I did have to make some decisions on which parts of the Othello-game to explore, as e.g. handcrafting features alone would have probably been enough to
  fill this project. Naturally, I decided to focus on the 'Deep' part. (18.11.2024)

## Remove before submission

notes on important aspects of the Othello Game: 
- auf versch. strategies trainieren: z.B. take as less / much pieces as possible
- sweet 16? bleibt das game eher in der mitte?
- äußerer Rand wird von mtte nach rand in B,A,C unterteilt.
- 1 rein auf beiden achsen richtung mitte sind X squares, die sind sehr wichtig anscheinend
- strategy: taking as less pieces as possible
- don't play the X squares -> von dort am ehesten der jump in die ecke
- gedankengang: mehrere flippen kann gut sein -> falls keine neuen moves
für den gegner entstehen
- maximizing no. of moves 
- change of strategies, zb mit minimizing starten und dann spätere
'facts' reintrainieren?
- szenarien/konkrete wichtige spielsituationen irgendwie spezifischer ins training einbauen?
WICHTIG: https://github.com/2Bear/othello-zero
Erkenntnisse:
- MinAgent vs MaxAgent:
  - Gewinner ist derjenige, der nicht beginnt! Immer. whut

- mu zero macht keinen Sinn hier, da bei othello keine unkown oder partially observable dynamics vorhanden sind
- 