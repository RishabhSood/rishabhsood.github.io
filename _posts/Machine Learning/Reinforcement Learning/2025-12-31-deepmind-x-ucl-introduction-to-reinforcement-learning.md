---
title: DeepMind x UCL | Introduction To Reinforcement Learning (2015)
date: 2025-12-31 10:05 +0530
categories: [Machine Learning, Reinforcement Learning]
tags: [machine_learning, reinforcement_learning, lecture_notes]     # TAG names should always be lowercase
math: true
toc: true
mermaid: true
description: Lecture Notes from David Silver's RL Series on youtube
---
[Watch the Lectures Here](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&si=fYkY_BOH0eJR7qye)
## Lecture 1: Introduction to Reinforcement Learning
[Official Lecture Slides](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/intro_rl.pdf)
> *RL is a fundamental science trying to understand the optimal way to make decisions.*
>
> *A large part of the human brain / behavior is linked to a dopamine system (reward system), this neurotransmitter dopamine reflects exactly on one of the main algorithms we apply in RL.*
{: .prompt-info }

### About
#### Characteristic of Reinforcement Learning

***Q. What makes reinforcement learning different from other Machine Learning Paradigms?***

- There is no supervisor only a `reward` signal
    - No direct indication of correct / wrong action, instead it’s a trial and error paradigm.
- Feedback is delayed not instantaneous.
    - Initial decisions may reap rewards many steps later.
    - Example: A decision may instantaneously look good, and reap a positive reward for the next few steps, but then plunge into a large catastrophic negative reward!
- Time really matters (sequential, not i.i.d data)
    - Step after step, the agent gets to make a decision / pick an action and see how much reward it gets, accordingly optimizing those rewards to get the best possible outcome.
    - Unlike the classical supervised / unsupervised system we’re not talking about utilizing an IID dataset and learning from it, we’re facing a dynamic system: an agent moving through a world.
- Agent’s actions affect the subsequent data it receives
    - What the agent sees at the next step would be highly correlated to what it sees at the current step, furthermore as the agent gets to take action to influence its environment, in other words the agent is itself influencing it’s rewards (active learning process)!

#### Examples of Reinforcement Learning

- Fly stunt manoeuvres in a helicopter
- [Defeat the world champion at Backgammon](https://www.bkgm.com/articles/tesauro/tdl.html)
- Manage an investment portfolio
- Control a power station
- Make a humanoid robot walk
- [Play many different Atari games better than humans](https://youtu.be/TmPfTpjtdgg?si=WQlpzl3jFo3ri6q8)

### The Reinforcement Learning Problem
#### Rewards
- A scalar / number feedback signal defined for every timestep `t` denoted by $R_t$, indicating how well the agent is doing at each timestep.
- The agent's goal / job is to maximise the cummulative sum of these feedback signals / rewards.
- *Reinforcement Learning is based on the **reward hypothesis***

> **Definition (Reward Hypothesis):**
>All goals can be described by the maximisation of expected cumulative reward.
{: .prompt-info }

> **Example Goal-Rewards** 
> - **Quickest to Target:** Reward is set as `-1` per timestep, hence the agent learns to take the least possible amount of steps.
> - **Fly Stunt manoeuvres in a trajectory:**
>   - `+ve` reward for following trajectory / large `-ve` reward for crashing
> - **Defeat the world champion at Backgammon:** `+/−ve` reward for winning/losing a game
> - **Manage an investment portfolio:** `+ve` reward for each $ in bank
> - **Control a power station:**
>   - `+ve` reward for producing power / large `-ve` reward for exceeding safety thresholds
> - **Make a humanoid robot walk:**
>   - `+ve` reward for forward motion / large `-ve` reward for falling over
> - **Play many different Atari games better than humans:** `+/−ve` reward for increasing/decreasing score
> - **No intermediate rewards:** We consider reward at the end of the episode to be the cummulative reward.
{: .prompt-tip }

#### Sequential Decision Making
> Although the examples we just discussed are all different from each other, we define a *unifying framework* for these by generalizing them to **Sequential Decision Making Processes** with the goal of *selecting **actions** to **maximize total future reward***.
{: .prompt-info }
- Actions may have long term consequences, Ergo Reward may be delayed.
- This also means that at times the agent might have to sacrifice immediate reward to gain more long-term reward (non-greedy).

> **Examples:** 
> - Blocking Opponent moves (might increase winning chances in future moves)
> - Refuelling the helicopter (to avoid a crash in future)
> - A financial investement (which would take a long period to mature)
{: .prompt-tip }

#### Agent and Environment
![agent receives and observation, performs an action and receives a reward](assets/posts/david_silver_rl/lec1_agent_n_env.png){: w="350"}


At each step $t$: 
- The agent (brain):
    - Receives an observation $O_t$
    - Receives a reward $R_t$.
    - Takes an action $A_t$ based off of the information that it's receiving from the world (observation $O_{t}$) and the reward signal it's getting ($R_{t}$).
- The environment:
    - Receives an action $A_t$
    - Emits observation $O_{t+1}$
    - Emits scalar reward $R_{t+1}$
- $t$ increments at env step.

Thus, the trial and error loop that we define for Reinforcement Learning is basically a time series of observations, rewards and actions, which in turn defines the experience of an agent (which is the data we then use for reinforcement leanring).

#### History & State
This stream of experience which comes in, i.e. the sequence of observations (all observable variables up till time `t` / incoming sensorimotor stream of the agent), actions & rewards is called `history`, and is denoted as:

$$H_t = O_1, R_1, A_1 \cdots, A_{t-1}, O_t, R_t$$

What happens next, then depends on this history:
- The agent selects an action 
- The environment signals a reward and emits an observation

The `history` in itself however is not useful specially for agents which have long lives, as with increased time their processing speed would linearly reduce, real time interactions would become impossible. To resolve this, we instead talk about something called `state`, which is like a summary of the information which determines what happens next.

Formally, state is defined as a function of the history:

$$S_t = f(H_t)$$

#### Definitions of State
##### <u>Environment State</u>
- The information that's used within the environment to decide what happens next, i.e. the environment's private representation: $S_t^e$. In other words whatever data the environment uses to pick the next observation/reward.
- This state is not usually visible to the agent. 

Hence, the environment state is more of a formalism, which helps us build the algorithms for our agent. Even if the environment state is visible, it might not directly contain relevant information for the agent to make good decisions.
##### <u>Agent State</u>
The agent state $S_t^a$ is a set of numbrs that capture exactly what's happened to the agent so far, summarize what's gone on, what it's seen so far and use those numbers to basically pick the next action. How we choose to process those observations and what to remember and what to throw away builds this state's representation. It can be any function of the history:

$$S_t^a = f(H_t)$$

This is the information (agent state) used by RL algorithms.
##### <u>Information / Markov State</u>
An information state (a.k.a Markov state) contains all useful information of the history.

> **Definition (Markov State):**
>A state $S_t$ is `markov` if and only if:
>
>$$\mathbb{P} [S_{t+1} | S_t] = \mathbb{P} [S_{t+1} | S_1, \cdots S_t]$$
>
>*in other words, you could throw away all previous states, as the current state in itself gets the same characterization (is a sufficient statistic) of the future.*
{: .prompt-info }

- The future is independent of the past, given the present:

$$H_{1:t} \rightarrow S_t \rightarrow H_{t+1:\infty}$$

- The environmnt state $S_t^e$ is Markov.
- The history $H_t$ is also Markov (although inefficient). Hence it's always possible to have a markov state, the question in practice is just how do we find that useful/efficient representation.

##### <u>Rat Example</u>
Consider the following classic, scientist-rat experiment scenario, say the rat observes these two sequences / experiences:
![S1: LT LT LV BL ZAP | S2: BL LT LV LV CH](assets/posts/david_silver_rl/lec1_rat_exepriences_observed.png)

Next, in the current sequence, the rat observes the following sequence of states:
![S3: LV LT LV BL ?](assets/posts/david_silver_rl/lec1_rat_new_exeprience.png)

The expected next state by the rat, then depends on it's representation of the state! For example:
- If the rat's state depends on the sequence of last 3 observed states, then the rat expects to be zapped!
- If the rat's state depends on the count of lever / light & bell, then it expects cheese!
- If the rat's state depends on complete sequence, that it doesn't know what to expect (not enough data)

#### Fully Observable Environments
In this specific scenario, the agent is able to directly observe the environment state:

$$O_t = S_t^a = S_t^e$$

- Agent State = Environment State = Information State
- Formally, this is called **Markov Decision Process (MDP)** (will be studied in the next lecture and the majority of this series).

#### Partially Observable Environments
In this specific scenario, the agent indirectly observes the environment:
- Examples: 
    - A robot with a camera stream trying to figure out where it is in the world.
    - A trading agent only observing current prices
    - A poker playing agent only observing public cards

$$\texttt{Agent State} \neq \texttt{Environment State}$$

- Formally, this is called **partially observable Markov Decision Process (POMDP)**.
- Agent must construct its own state representation in this case $S_t^a$, ex:
    - Complete history: $S_t^a = H_t$
    - Beliefs of Environment State: $$S_t^a = (\mathbb{P}[S_t^e = s^1], \cdots, \mathbb{P}[S_t^e = s^n])$$
    - Recurrent Neural Network: $$S_t^a = \sigma(W_s \cdot S_{t-1}^a + W_o \cdot O_t)$$

### Inside an RL Agent
An RL agent may include one or more of these components (not exhaustive ofcourse!):
- **Policy:** agent’s behaviour function (how the agent decides what action to take when in a given state)
- **Value function:** how good is each state and/or action (how much reward do we expect if we take an action from a given state). In other words, estimating how well we are doing in a particular situation.
- **Model:** agent’s representation of the environment (how the agent thinks the environment works)

#### Policy
A policy is the agent's behavior, it is a map from state to action, which may or may not be deterministic:
- **Deterministic:** $\pi(s) = a$
- **Stochastic:** $\pi(a \mid s) = \mathbb{P} [A=a \mid S=s]$

#### Value Function
The value function is a prediction of expected future reward, used to evaluate goodness/baddness of states, hence helping with action choice. Written as:

$$v_\pi(S) = \mathbb{E}_\pi[R_t + \gamma \cdot R_{t+1} + \gamma^2 \cdot R_{t+2} \cdots \mid s = S]$$

*The behavior that maximizes this value, is the one that correctly trades off agent's risk so as to get the maximum amount of reward going into the future.*
(Notice that the value function depends on the policy chosen $\pi$, here $\gamma \in [0,1]$ is the discount factor that determines how much we value future rewards vs immediate rewards)

> Example: DQN Value Function in Atari [Video Lecture](https://youtu.be/2pWv7GOvuf0?si=71ljntBYOgoGzhRi&t=3732) (1:02:12 - 1:04:36)
{: .prompt-tip }

#### Model
A model is basically not the environment itself but an estimate of what the environment might do. Hence, the agent tries and learns the behavior of the environment and then uses that model of the environment to help make a plan to help figure out what to do next. Generally, this model has two parts:
- **Transitions:** $\mathcal{P}$ predicts the next state (i.e. dynamics)
    - $$\mathcal{P}_{ss'}^{a} = \mathbb{P}[S' = s' \mid S = s, A = a]$$
- **Rewards:** $\mathcal{R}$ predicts the next (immediate) reward, e.g.
    - $$\mathcal{R}_{s}^{a} = \mathbb{E}[R \mid S = s, A = a]$$

It's however not always required / necessary to build the environment's model, in fact most of the lectures in this series will focus on model free methods instead.

#### Maze Example
![Maze example showing grid layout, policy arrows, value function, and learned model](assets/posts/david_silver_rl/lec1_maze_example.png){: w="350"}

In the above maze example, consider the following environment dynamics:
- Reward: -1 per time-step
- Actions: N E S W
- States: Agent's location

- Fig. A represents the layout of the maze
- Fig. B represents the optimal deterministic policy mapping for each state,
- Fig. C represents value $v_\pi(s)$ for each state $s$
- Fig. D represents the agent's internal environment model, after it traverses a path (might be imperfect):
    - The grid layout represents transition model $\mathcal{P}_{ss'}^{a}$
    - The numbers in each model state represent immediate reward $\mathcal{R}_{s}^{a}$ for each state $s$ (same for all $a$)

#### Categorizing RL Agents
- Taxonomy 1:
    - Value Based
        - No Policy (implicit) - Optimal Actions can be read out greedily based on value function
        - Value Function
    - Policy Based
        - Policy
        - No Value Function
    - Actor Critic
        - Policy
        - Value Function 
- Taxonomy 2:
    - Model Free
        - Policy and/or value function
        - No Model
    - Model Based
        - Policy and/or value function
        - Model

### Problems within Reinforcement Learning
#### Learning and Planning
Two fundamental problems in sequential decision making
- Reinforcement Learning:
    - The environment is initially unknown
    - The agent interacts with the environment
    - The agent improves its policy
    - Example (Atari): Learn to play a game from itneractive-gameplay by picking actions on joystick and observing pixels and scores.
- Planning:
    - A model of the environment is known
    - The agent performs computations with its model (without any external interaction)
    - The agent improves its policy
    - a.k.a. deliberation, reasoning, introspection, pondering, thought, search
    - Example (Atari): A queryable emulator ready at hand, which responds back with the next state and score based on current state and action (rules of the game are known). Using this model to plan ahead and find the optimal policy (ex: Tree Search).

#### Exploration and Exploitation
Reinforcement learning is like trial-and-error learning, we expect the agent to discover a good policy from its experiences of the environment, without losing too much reward along the way.
- **Exploration** finds more information about the environment, by possibly ignoring existing best-known reward policy
- **Exploitation** exploits known information to maximise reward
- It is usually important to explore as well as exploit

> Examples:
> - Restaurant Selection
>   - **Exploitation** Go to your favourite restaurant
>   - **Exploration** Try a new restaurant
> - Online Banner Advertisements
>   - **Exploitation** Show the most successful advert
>   - **Exploration** Show a different advert
{: .prompt-tip}

#### Prediction and Control
- **Prediction:** How well will the agent do if it follows a given policy. (value function prediction)
- **Control:** Finding the optimal policy to get the most reward
- In RL we typically need to solve the Prediction problem in order to solve the Control Problem, i.e. we evaluate all of our policies to figure out the best.

![Prediction and Control Example](assets/posts/david_silver_rl/lec1_prediction_n_control.png){: w="350"}
> Example: Going back to our grid world with a -1 reward, we can possibly ask the following questions:
> - Prediction: If we choose to randomly move across the grid, what's our expected reward from a given state?
> - Control: What's the optimal behavior on the grid if I am in a given state? What's the value function if I follow the optimal behavior?
{: .prompt-tip}

## Lecture 2: Markov Decision Process
[Official Lecture Slides](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-2-mdp.pdf)

### Markov Processes
### Markov Reward Processes
### Markov Decision Processes
### Extension to MDPs