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

Note: 8x8 Othello is claimed to be solved by a preprint from 2023. Otto will to be trained starting on a 8x8 configuration, later on a 10x10 or potentially even bigger configuration.

Otto's goal is to beat the current performances of Othello-bots, with respect to the computational cost. (Bring you own method)

The following papers seem as an interesting starting point into this subject:
- [OLIVAW: Mastering Othello without Human Knowledge, nor a Fortune](https://arxiv.org/abs/2103.17228). This paper from 2021 aims to minimze the computational power needed to achieve the performance of the state of the art, which covers the aspect of keeping computational cost as low as possible.
- [Deep reinforcement Learning Using Monte-Carlo Tree Search for Hex and Othello](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2777474).
- [Hierarchical Reinforcement Learning for the Game of Othello](https://ir.canterbury.ac.nz/bitstreams/51750833-63c1-42a6-b7bf-08ecd4c58434/download)
- [Q-learning adaptations in the game Othello](https://fse.studenttheses.ub.rug.nl/id/eprint/23027)

Another interesting link on this topic:
- [GymOthelloEnv](https://github.com/lerrytang/GymOthelloEnv)

Data of real played othello games are rather easy to find on the web, e.g. [https://www.kaggle.com/datasets/andrefpoliveira/othello-games](https://www.kaggle.com/datasets/andrefpoliveira/othello-games).

However, three different approaches will be implemented:
- train on human-generated games only
- train on simulated games only
- train on human-generated games, afterwards on simulated ones


## Intended work-packages

- Read through papers above to fully grasp state-of-the-art-approaches (10h)
- Set up infrastructure (15h). DONE: Simple Board-class, printing in the console and test to validate correctness took around 5h.
- Implement SOFA (Q-learning) (15h)
- Implement the approaches mentioned above (10h each)
- Finetune Neural Net, try out different architectures (35h)
- Using an existing application to showcase the results (5h)
- Final report and presentation (5h)


notes: 
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
