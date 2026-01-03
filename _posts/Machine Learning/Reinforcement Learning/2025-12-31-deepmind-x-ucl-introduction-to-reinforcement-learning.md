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
>
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
##### <span style="color: grey;"><u>Environment State</u></span>
- The information that's used within the environment to decide what happens next, i.e. the environment's private representation: $S_t^e$. In other words whatever data the environment uses to pick the next observation/reward.
- This state is not usually visible to the agent. 

Hence, the environment state is more of a formalism, which helps us build the algorithms for our agent. Even if the environment state is visible, it might not directly contain relevant information for the agent to make good decisions.
##### <span style="color: grey;"><u>Agent State</u></span>
The agent state $S_t^a$ is a set of numbrs that capture exactly what's happened to the agent so far, summarize what's gone on, what it's seen so far and use those numbers to basically pick the next action. How we choose to process those observations and what to remember and what to throw away builds this state's representation. It can be any function of the history:

$$S_t^a = f(H_t)$$

This is the information (agent state) used by RL algorithms.
##### <span style="color: grey;"><u>Information / Markov State</u></span>
An information state (a.k.a Markov state) contains all useful information of the history.

> **Definition (Markov State):**
>
>A state $S_t$ is `markov` if and only if:
>
>$$\color{pink}\mathbb{P} [S_{t+1} | S_t] = \mathbb{P} [S_{t+1} | S_1, \cdots S_t]$$
>
>*in other words, you could throw away all previous states, as the current state in itself gets the same characterization (is a sufficient statistic) of the future.*
{: .prompt-info }

- The future is independent of the past, given the present:

$$H_{1:t} \rightarrow S_t \rightarrow H_{t+1:\infty}$$

- The environmnt state $S_t^e$ is Markov.
- The history $H_t$ is also Markov (although inefficient). Hence it's always possible to have a markov state, the question in practice is just how do we find that useful/efficient representation.

##### <span style="color: grey;"><u>Rat Example</u></span>
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

### Markov Processes / Chain
#### Introduction
##### <span style="color: grey;"><u> Introduction to MDPs </u></span>
Markov Decision Processes are a formal definition of the environment that the agent interacts with, applying only to the cases where the [environment is fully observable](#fully-observable-environments).

In other words, the current state of the environment given to the agent completely characterizes this process (the way in which the environment unfolds depends on it). Almost all RL Problems an be formalized as MDPs, examples:
- *Optimal control* primarily deals with <i>continuous MDPs</i>
- *Partially observable problems* can be converted into MDPs
- *Bandits* are MDPs with one state (an agent is given a choice of actions, it chooses one.)

#### Markov Process
##### [Markov Property](#information--markov-state)
*(already covered in Lecture 1, click on the shortcut above to refresh the concept!)*

##### <span style="color: grey;"><u>State Transition Matrix</u></span>
Given a markov state $s$ and a successor state $s'$, the *state transition probability is defined by:

$$\mathcal{P}_{ss'} = \mathbb{P}[S_{t+1} = s' \mid S_t = s]$$

The state transition matrix $\mathcal{P}$, then defines transition probabilities from all states $s$ to all successor states $s'$.

$$
\mathcal{P} = \text{from } \overset{\text{to}}{
\begin{bmatrix}
\mathcal{P}_{11} & \dots & \mathcal{P}_{1n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{n1} & \dots & \mathcal{P}_{nn}
\end{bmatrix}
}
$$

Where, by the law of probability, each row sums to 1.

##### <span style="color: grey;"><u>Markov Process / Chain</u></span>
A Markov process is a memoryless random process, i.e. a sequence of random states $S1, S2, \cdots$ with the Markov property.
> **Definition (Markov Process):**
>
>A *Markov Process (or Markov Chain)* is a tuple $\langle\textcolor{pink}{\mathbfcal{S}, \mathbfcal{P}}\rangle$, where:
>- $\color{pink}\mathcal{S}$ is a (finite) set of states
>- $\color{pink}\mathcal{P}$ is a state transition probability matrix,
>
>   $$\color{pink}\mathcal{P}_{ss'} = \mathbb{P}[S_{t+1} = s' \mid S_t = s]$$
{: .prompt-info }

##### <span style="color: grey;"><u>Example: Student Markov Chain</u></span>
![Student Markov Process Example from Lecture](assets/posts/david_silver_rl/lec2_markov_process_example.png){: w="350"}

Notice that, "Sleep" is a terminal state which indefinitely keeps looping back to itself. Based on the above example chain, we can draw out sample **episodes**, starting from say $S_1 = C_1$:
- $C1, C2, C3, Pass, Sleep$
- $C1, FB, FB, C1, C2, Sleep$
- $C1, C2, C3, Pub, C2, C3, Pass, Sleep$
- $C1, FB, FB, C1, C2, C3, Pub, C1, FB, FB, FB, C1, C2, C3, Pub, C2, Sleep$

The above episodes are Random sequences that are drawn from a probability distribution over sequences of states and the fact it has the markov property means that it can be described by one of these diagrams. It can be described by saying from any state there's some probability of transitioning to any other state.

$$
\mathcal{P} =
\begin{array}{r}
\begin{array}{ccccccc} \small{C1} & \small{C2} & \small{C3} & \small{Pass} & \small{Pub} & \small{FB} & \small{Sleep} \end{array} \\
\begin{array}{r} \small{C1} \\ \small{C2} \\ \small{C3} \\ \small{Pass} \\ \small{Pub} \\ \small{FB} \\ \small{Sleep} \end{array}
\left[
\begin{array}{ccccccc}
 & 0.5 & & & & 0.5 & \\
 & & 0.8 & & & & 0.2 \\
 & & & 0.6 & 0.4 & & \\
 & & & & & & 1.0 \\
0.2 & 0.4 & 0.4 & & & & \\
0.1 & & & & & 0.9 & \\
 & & & & & & 1
\end{array}
\right]
\end{array}
$$

The above state transition matrix, summarizes the transition probabilities of the defined chain.

> Q. How do we deal with modifications of these probabilities over time? Ex: Say that the probability of revisiting facebook decreases over time.
>
> Ans. So there's two answers to that:
> - One answer is that you can have non-stationary Markov process, later we'll have mdp so non-stationary mdp, in that case what you can do is use the same kind of algorithms we use for the stationary case but incrementally adjust your solution algorithm to just kind of track the best solution you found so far
> - The other answer is to say well the fact you've got non-stationary Dynamics just makes it a more complicated markup process. Imagining for this scenario, that there's probabilities that depend on how long you've been in this state and so forth so now you can ugment this state you can have a more complicated Mark process that has like a counter that tells you how long you've been in this Facebook State and now you've basically got lots of different states depending on whether you've been in Facebook once or twice or three times, ergo you can have an infinite / continuous set of these.
>
> All of these things are possible but they don't change the fundamental structure from this being a markov process this is just a particular simple instantiation where we can see everything on one page but don't let that mislead you into thinking that markov processes are necessarily small or simple, we will solve very very complex markov processes in the rest of the course. We'll see how to solve you know mdps sometime some of them have $10^{170}$ States. So you know, these could be very large and complex beasts with a lot of structure to them.
{: .prompt-tip }

### Markov Reward Processes (Markov Chain + Rewards)
#### Introduction
A Markov Reward Process is a Markov Chain with value judgements (saying how much reward will the agent have accumulated across some particular sequence that we sampled from a given Markov process).

> **Definition (Markov Reward Process):**
>
>A *Markov Reward Process* is a tuple $\langle\mathcal{S}, \mathcal{P}, \textcolor{pink}{\mathbfcal{R}, \boldsymbol\gamma}\rangle$, where:
>- $\mathcal{S}$ is a (finite) set of states
>- $\mathcal{P}$ is a state transition probability matrix,
>
>   $$\mathcal{P}_{ss'} = \mathbb{P}[S_{t+1} = s' \mid S_t = s]$$
> - $\textcolor{pink}{\mathbfcal{R}}$ is a reward function, $\mathcal{R_s} = \mathbb{E}[\mathcal{R}_{t+1} \mid \mathcal{S}_t = s]$
> - $\textcolor{pink}{\boldsymbol\gamma}$ is a discount factor, $γ \in [0, 1]$
{: .prompt-info }

#### Example: Student Markov Reward Process
![Student Markov Reward Process Example from Lecture](assets/posts/david_silver_rl/lec2_makov_rp_example.png){: w="350"}

#### Return
> **Definition (Return):**
>
> For a given sample experience drawn from the chain, the return $\textcolor{pink}{\mathbf{G_t}}$ is defined as:
> $$\color{pink}G_t = R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} \cdots = \sum_{k=1}^{\infty}\gamma^k\cdot R_{t+1+k}$$
>
> The discount factor $\gamma$ (how much we care *now* for the rewards we get in the *future*) weighs immediate reward over delayed reward:
> - $\gamma$ close to 0 leads to: *myopic evaluation*
> - $\gamma$ close to 1 leads to: *farisghted evaluation*
{: .prompt-info }

##### <span style="color: grey;"><u>Why Discount?</u></span>
- It's Mathematically convenient to discount rewards, avoids infinite returns in cyclic Markov processes too
- Uncertainty about the future may not be fully represented (i.e. we are uncertain about the accuracy of our model, and hence choose to trust later predicted rewards less)
- Animal/human behaviour shows preference for immediate reward (ex: financial rewards).

Although, It is sometimes possible to use *undiscounted Markov reward processes* (i.e. $\gamma=1$), e.g. if all sequences terminate.

> With the average reward formulation, it's possible even with infinite sequences to still deal with the undiscounted evaluation of markov processes processes and mdps. (Discussed in extend notes of this class)
#### Value Function
The value function $v(s)$ gives the long-term value of state s (i.e. the expected return if the agent starts an episode from state s).
> **Definition (Value Function):**
>
> The state value function v(s) of an MRP is the expected return starting from state s:
> 
> $$v(s) = \mathbb{E}[G_t \mid S_t = s]$$
{: .prompt-info}
##### <span style="color: grey;"><u>Example</u></span>
Sample returns for Student MRP, Starting from $S_1 = C_1$ with $\gamma=\frac{1}{2}$:
![Return Value Example from the lecture](assets/posts/david_silver_rl/lec2_return_value_example.png)
(Although the selected episodes for which the returns are calculated are random, but the calculated values for these states individually (next figure) won't be a random quantity, as it's an expectation over these random quantities.)
![Value Function Example from the lecture](assets/posts/david_silver_rl/lec2_value_function_example.png)

#### Bellman Equation for MRPs
The value function obeys a recursive decomposition consisting of the following two parts:
- immediate reward $R_{t+1}$
- Discounted value of the successor state: $\gamma \cdot G_{t+1}$

$$
\begin{aligned}
v(s) &= \mathbb{E}[G_t \mid S_t = s] \\
     &= \mathbb{E}[R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \cdots \mid S_t = s] \\
     &= \mathbb{E}[R_{t+1} + \gamma \cdot (R_{t+2} + \gamma R_{t+3} + \dots) \mid S_t = s] \\
     &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
     &= \mathbb{E}[R_{t+1} + \gamma \cdot v(S_{t+1}) \mid S_t = s]
\end{aligned}
$$

> (The transition from the second last to last step comes from the law of expected iterations, which states that the overall expected value of a random variable $X$, $\mathbb{E}[X]$, can be found by first taking the conditional expectation of $X$ given another variable $Y$, $\mathbb{E}[X\mid Y]$, and then averaging these conditional expectations over all possible values of $Y$. Mathematically, this is expressed as $E[X]=E[E[X\mid Y]]$)

We do one-step lookaheads, by moving one step ahead from current state, take an average of all possible outcomes (with transition probabilities) together, which gives us the value function at the current step.
$$v(s) = \mathcal{R}_s + \sum_{s' \in S} \mathcal{P}_{ss'}\cdot v(s')$$

Here is a backup diagram ([what's a backup diagram?](https://towardsdatascience.com/all-about-backup-diagram-fefb25aaf804/)) of this process:
![Value Function Backup Diagram](assets/posts/david_silver_rl/lec2_value_function_backup_diagram.png){: w="350"}

##### <span style="color: grey;"><u>Student MRP Example</u></span>
The below figure shows equivalence of the value at a given state to the immediate reward and a discounted (just 1 in this case) average of lookaheads for a converged value function for the Student MRP with a $\gamma$ factor of 1:
![Bellman Equation For Student MRP](assets/posts/david_silver_rl/lec2_mrp_bellman_equation_example.png){: w="350"}

##### <span style="color: grey;"><u>Matrix Representation</u></span>
$$\left[\begin{matrix}v(1)\\\vdots\\v(n)\end{matrix}\right] = \left[\begin{matrix}\mathcal{R}_1\\\vdots\\\mathcal{R}_n\end{matrix}\right] + \gamma \begin{bmatrix}
\mathcal{P}_{11} & \dots & \mathcal{P}_{1n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{n1} & \dots & \mathcal{P}_{nn}
\end{bmatrix} \left[\begin{matrix}v(1)\\\vdots\\v(n)\end{matrix}\right]$$

#### Solving the Bellman Equation
As can be seen, the bellman equation is linear and can be solved directly:

$$
\begin{aligned}
v &= \mathcal{R} + \gamma \cdot \mathcal{P}v \\
     &= (1 - \gamma \cdot \mathcal{P})^{-1} \mathcal{R} \\
\end{aligned}
$$

However, the time-complexity for this borders around $n^3$ (due to the inverse calculation), and hence it practically only applies to small MRPs. For larger MRPs, we go for other iterative solutions such as:
- Dynamic programming
- Monte-Carlo evaluation
- Temporal-Difference learning

> Another point to note is that this nice linear property goes away, once we move to MDPs, where instead of evaluating Rewards, we are looking for max possible rewards for each state, which makes the above equation more complex.

### Markov Decision Processes (Markov Chain + Rewards + Actions)
#### Introduction
A Markov decision process (MDP) is a Markov reward process with decisions. It is an environment in which all states are Markov.

> **Definition (Markov Decision Process):**
>
>A *Markov Decision Process* is a tuple $\langle\mathcal{S}, \textcolor{pink}{\mathbfcal{A}}, \mathcal{P}, \mathbfcal{R}, \boldsymbol\gamma\rangle$, where:
>- $\mathcal{S}$ is a (finite) set of states
>- $\textcolor{pink}{\mathbfcal{A}}$ is a (finite) set of Actions
>- $\mathcal{P}$ is a state transition probability matrix,
>
>   $$\mathcal{P}_{ss'}^\textcolor{pink}{a} = \mathbb{P}[S_{t+1} = s' \mid S_t = s, \textcolor{pink}{A_t = a}]$$ (Notice that the transition probability from one state to another may depend on the selected action too now)
> - $\mathbfcal{R}$ is a reward function, $$\mathcal{R_s^\textcolor{pink}{a}} = \mathbb{E}[\mathcal{R}_{t+1} \mid S_t = s,\textcolor{pink}{A_t = s}]$$ (Notice that the reward may depend on the selected action too now)
> - $\boldsymbol\gamma$ is a discount factor, $γ \in [0, 1]$
{: .prompt-info }

#### Example: Student Markov Decision Process
![Student MDP Example from the lecture](assets/posts/david_silver_rl/lec2_student_mdp_example.png){: w="350"}

Notice now that, there are no probabilities attached to probable transitions from one state to the other, that is we're free to move about the states (take decisions / make choices), each transition now has a dedicated reward.

Notice that, post taking the action the agent doesn't always deterministically land on a state, the environment may influence which state the agent lands in. As an example consider that the agent lands in the class 3 node and chooses the "Pub" action, thereby gaining a reward of $-2$, next it depends on the environment to which node it wishes to push the agent to, in the current scenario, the environment pushes the agent to class 1 with probability 0.2 & classes 2 & 3 with probability 0.4 each.

There's a lot more control that the agent can exert now over its path through this mdp. This is the reality for an agent and the goal / the game we'll be playing now is to try and find the ***best path** through our decision making process that **maximizes the sum of rewards** that the agent gets*.

#### Policies
> **Definition (Policy):**
>
>A policy $\pi$ is a distribution over actions given states,
>
>$$\pi(a \mid s) = \mathbb{P}\left[A_t = a \mid S_t = s\right]$$
{: .prompt-info }

- A policy fully defines the behavior of the agent
- MDP policies depend only on the current state (not the history), in other words, policies are stationary:
    $$A_t = \pi(.|S_t), \forall t \gt 0$$
- It's useful to make the policy stochastic because that allows us to do things like exploration (more on this later in the lectures)
- We can always recover a **Markov Process** or a **Markov Reward Process** from a **Markov Decision Process**. Given an MDP $$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$ and a policy $\pi$, we can define a:
    - **MP** from a **MDP** as $\Rightarrow$ $\langle \mathcal{S}, \mathcal{P^\pi} \rangle$ for a state sequence: $S_1, S_2, S_3, \cdots$
    - **MRP** from a **MDP** as $\Rightarrow$ $\langle \mathcal{S}, \mathcal{P^\pi}, \mathcal{R^\pi}, \gamma \rangle$ for a state-reward sequence $S_1, R_2, S_2, \cdots$
    - Where:
        - $\mathcal{P_{ss'}^\pi} = \sum_{a \in A}\pi(a\mid s) \mathcal{P_{ss'}^a}$
        - $\mathcal{R_{s}^\pi} = \sum_{a \in A}\pi(a\mid s) \mathcal{R_{s}^a}$

#### Value Function
> **Definition (Value Function):**
>
>The state-value function $v_{\pi}(s)$ of an MDP is the expected return starting from state $s$, and then following policy $\pi$
>
>$$v_{\pi}(s) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s\right]$$
{: .prompt-info }

> **Definition (Action-Value Function):**
>
>The action-value function $q_{\pi}(s,a)$ of an MDP is the expected return starting from state $s$, taking action $a$ and then following policy $\pi$
>
>$$q_{\pi}(s,a) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s, A_t = a\right]$$
{: .prompt-info }

> Note that $\mathbb{E}_{\pi}$ indicates, expected value given that we're sampling from the MDP while following policy $\pi$

##### <span style="color: grey;"><u>Example: State-Value Function for Student MDP</u></span>
![Lecture Example for State-Value Function for Student MDP](assets/posts/david_silver_rl/lec2_state-value_fn_mdp_ex.png){: w="350"}

#### Bellman Expectation Equation
The state-value function can again be decomposed into immediate reward plus discounted value of successor state:

$$v_{\pi}(s) = \mathbb{E}\left[R_{t+1} + \gamma\cdot v_{\pi}(S_{t+1}) \mid S_t = s\right]$$

Similarly for the action-value function:

$$q_{\pi}(s, a) = \mathbb{E}\left[R_{t+1} + \gamma\cdot q_{\pi}(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a\right]$$

Now, consider the following backup diagram, which indicates the value function at a given state. From a given state, we choose actions, and when following a policy $\pi$, we choose these actions with a probability of $\pi(a\mid s)$. For each choosen action from a given state, we have an expected reward from there-on defined by the action value function: $q_{\pi}(s, a)$. Thus, we get:

![Bellman Expectation Equation Backup Diagram linking state-value to action-value](assets/posts/david_silver_rl/lec2_vq_backup_dg_mdp.png){: width="350"}

$$v_{\pi}(s) = \sum\limits_{a \in \mathcal{A}}\pi(a\mid S) q_{\pi}(s, a)$$

Following the previous chain of thought, now consider the fact that we have chosen a given action from a state, which directly gives us the immediate reward. Next, depending on the stochasticity of the environment, we may now landup in one of the many probable states post taking an action, as we already know the probability of this is given by: $\mathcal{P_{ss'}^a}$. Hence, we can write the action-value function at a given action choice from a state as:
![Bellman Expectation Equation Backup Diagram linking action-value to state-value at next step](assets/posts/david_silver_rl/lec2_qv_backup_dg_mdp.png){: width="350"}

$$q_{\pi}(s, a) = R_{s}^a + \gamma \sum\limits_{s' \in \mathcal{S}}\mathcal{P_{ss'}^a} v_{\pi}(s')$$

> Notice that, the expected reward coming from the second term is from next step and thus discounted.

##### <span style="color: grey;"><u>Bellman Expectation Equation for $v_{\pi}(s)$</u></span>
Combining the above 2 observations, we can now write a recursive relation for the state-value function as:
![Bellman Expectation Equation Backup Diagram for State Value Function in MDP](assets/posts/david_silver_rl/lec2_bellman_ee_state_value.png){: width="350"}

$$v_{\pi}(s) = \sum\limits_{a \in \mathcal{A}}\pi(a\mid S)(R_{s}^a + \gamma \sum\limits_{s' \in \mathcal{S}}\mathcal{P_{ss'}^a} v_{\pi}(s'))$$

##### <span style="color: grey;"><u>Bellman Expectation Equation for $q_{\pi}(s,a)$</u></span>
Similarly, we can now write a recursive relation for the action-value function as:
![Bellman Expectation Equation Backup Diagram for Action Value Function in MDP](assets/posts/david_silver_rl/lec2_bellman_ee_action_value.png){: width="350"}

$$q_{\pi}(s, a) = R_{s}^a + \gamma \sum\limits_{s' \in \mathcal{S}}\mathcal{P_{ss'}^a}(\sum\limits_{a' \in \mathcal{A}}\pi(a'\mid s') q_{\pi}(s', a'))$$

##### <span style="color: grey;"><u>Example: Bellman Expectation Equation for Student MDP</u></span>
![Bellman Expectation Equation for Student MDP Example](assets/posts/david_silver_rl/lec2_bellman_expectation_eqn_student_mdp.png){: width="350"}

The above example, illustrates verification of the converged value for the state-value function at a given state (Class 3), such that $\pi(a \mid s) = 0.5$ & $\gamma = 1$.

##### <span style="color: grey;"><u>Matrix Representation</u></span>
Alternatively, we can represent the state-value function solution via matrix representation by converting our MDP to MRP, as discussed earlier [here](#solving-the-bellman-equation) and [here](#policies):

$$
\begin{aligned}
v_{\pi} &= \mathcal{R^{\pi}} + \gamma \cdot \mathcal{P^{\pi}}v_{\pi} \\
     &= (1 - \gamma \cdot \mathcal{P^{\pi}})^{-1} \mathcal{R^{\pi}} \\
\end{aligned}
$$

#### Optimal Value Function
> **Definition (Optimal Value Function):**
>
>The optimal state-value function $v_{*}(s)$ is the maximum value function over all policies:
>
>$$v_{*}(s) = \underset{\pi}{\text{max}} \quad v_{\pi}(s)$$
>
>The optimal action-value function $q_{*}(s, a)$ is the maximum action-value function over all policies
>
>$$q_{*}(s, a) = \underset{\pi}{\text{max}} \quad q_{\pi}(s, a)$$
{: .prompt-info }

- The optimal value function specifies the best possible performance in the MDP.
- An MDP is solved, when we know the optimal value function.

> To study in depth about conditions under which Markov Processes are well-defined refer to Dynamic Programming and Optimal Control (Volume II) by Dimitri P. Bertsekas.

##### <span style="color: grey;"><u>Example: Optimal Value Functions for Student MDP</u></span>
![Optimal State & Value Functions for Student MDP example](assets/posts/david_silver_rl/l2_optimal_sna-value-function-student-mdp.png)

#### Optimal Policy
We define a partial ordering over policies, by:

$$\pi \ge \pi^*\quad\text{if}\quad v_{\pi}(s) \ge v_{\pi^*}(s) \forall s$$

> **Theorem:**
>
>For any Markov Decision Process:
>- There exists altleast one optimal policy $$\pi^{*}$$ that is better than or equal to all other policies:
>
>   $$\pi^* \ge \pi \forall \pi$$
>
>- All optimal policies achieve the optimal value function:
>
>   $$v_{\pi^*}(s) = v_{*}(s)$$
>
>- All optimal policies achieve the optimal action-value function,
>
>   $$q_{\pi^*}(s, a) = q_{*}(s, a)$$
{: .prompt-warning }

##### <span style="color: grey;"><u>Finding an Optimal Policy</u></span>
An optimal policy can be found by maximising over $q_{*}(s,a)$:

$$
\pi_{*}(a|s) = \begin{cases}
1, & \text{if } a = \underset{a \in A}{\text{argmax}} \ q_{*}(s,a) \\
0, & \text{else } \\
\end{cases}
$$

- There is always an optimal deterministic policy for any MDP
- If we know $q_{*}(s,a)$, we immediately have the optimal policy

##### <span style="color: grey;"><u>Optimal Policy for student MDP</u></span>
![Optimal Policy for Student MDP](assets/posts/david_silver_rl/lec2_optimal_policy_for_student_mdp.png){: width="450"}

#### Bellman Optimality Equation
The optimal value functions are recursively related by the Bellman optimality equations:
- Consider the following backup diagram, which indicates the optimal state-value function at a given state (notice that unlike the expectation equation, we take the max and not weighted average!)

    ![Bellman Optimality Equation for State Value Function](assets/posts/david_silver_rl/lec2_bellman_optimality_eqn_state_value.png){: width="350"}

    $$v_*(s) = \underset{a \in A}{\text{max}} \ q_*(s, a)$$

- Consider the following backup diagram, which indicates the optimal action-value function at a given state (we can't take a max here, as the state transition probabilites are not in our control! they're dependent on the environment)

    ![Bellman Optimality Equation for Action Value Function](assets/posts/david_silver_rl/lec2_bellman_optimality_eqn_action_value.png){: width="350"}

    $$q_*(s, a) = \mathcal{R}_s^a + \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_*(s')$$

##### <span style="color: grey;"><u>For $v_*$</u></span>
Combining the above 2 observations, we can now write a recursive relation for the optimal state-value function as:

![Optimal State Value Function, Recursive Relation Backup Diagram](assets/posts/david_silver_rl/lec2_bellman_optimality_eqn_state_value_recursive.png){: width="350"}

$$v_*(s) = \underset{a \in A}{\text{max}} \ (\mathcal{R}_s^a + \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} v_*(s'))$$

Here's an example from the Student MDP, illustrating this relation:
![Optimal State Value Function, Recursive Relation Student MDP Example](assets/posts/david_silver_rl/lec2_bellman_optimality_eqn_state_value_rec_student_mdp.png){: width="450"}

##### <span style="color: grey;"><u>For $Q^*$</u></span>
Similarly, we can now write a recursive relation for the optimal action-value function as:

![Optimal Action Value Function, Recursive Relation Backup Diagram](assets/posts/david_silver_rl/lec2_bellman_optimality_eqn_action_value_recursive.png){: width="350"}

$$q_*(s, a) = \mathcal{R}_s^a + \sum_{s' \in \mathcal{S}} \mathcal{P_{ss'}^a} \ \underset{a' \in A}{\text{max}} \ q_*(s', a')$$

#### Solving the Bellman Optimality Equation
- Bellman Optimality Equation is non-linear
- No closed form solution (in general)
- Many iterative solution methods
    - Value Iteration
    - Policy Iteration
    - Q-learning
    - Sarsa

### FAQs

- **Q1.** *How can uncertainty and model imperfection be handled in Markov Decision Processes (MDPs)?*
    
    Ans. An MDP is only an approximate model of a real-world environment, so uncertainty is unavoidable. There are several ways to represent and manage this imperfection:
	1.	Bayesian / Explicit Uncertainty Modeling
    - Maintain a posterior distribution over possible MDP dynamics.
    - Solve for a policy that is optimal across a distribution of MDPs, rather than a single one.
    - This approach is principled but computationally very expensive.
	2.	Augmenting the State to Implicitly Encode Uncertainty
    - Incorporate uncertainty directly into the state representation.
    - For example, a state may include:
        - The agent’s physical configuration, and
        - Beliefs about the environment’s dynamics, or
        - The history of observations so far.
    - This avoids explicit reasoning over all uncertainties and is often more tractable, though less intuitive since uncertainty is not made explicit.
	3.	Using Discount Factors to Reflect Model Imperfection
    - Accept that the model is imperfect and encode uncertainty via the discount factor (γ).
    - A lower discount factor reflects greater uncertainty about future outcomes.
    - The discount factor can even be state-dependent, allowing some states to be modeled as more uncertain than others.

- **Q2.** *Do standard MDP formulations ignore risk and variance, and how can risk-sensitive objectives be handled?*

    Ans: Classical MDPs focus on maximizing expected return, without explicitly accounting for risk or variance in outcomes. However, this does not mean that risk cannot be modeled. There are two main perspectives:
    1. Transforming a Risk-Sensitive Problem into a Standard MDP
    - Any risk-sensitive objective (e.g., penalizing variance in returns) can, in principle, be transformed into an equivalent MDP with a modified reward function.
    - This new reward implicitly incorporates risk (such as variance-based costs) into the final return.
    - As a result, standard expectation-maximizing RL methods can still be applied.
    - The downside is that this transformation may require:
        - Augmented state representations (e.g., remembering past observations or returns), and
        - Solving a much more complex MDP.
    2.	Explicitly Studying Risk-Sensitive MDPs
    - There is a substantial body of research on risk-sensitive reinforcement learning, where policies are optimized not just for expected return, but also for:
        - Variance of returns,
        - Downside risk, or
        - Other risk measures.
    - These approaches go beyond expectation-based optimization and directly incorporate risk into the objective.

    In summary, while standard MDPs appear risk-neutral, risk sensitivity can either be embedded implicitly via reward transformations or addressed explicitly through specialized risk-sensitive MDP formulations, both of which are active areas of research.

- **Q3.** *What is the intuition behind the Bellman optimality equation in a real-world example?*

  **Ans.** The Bellman optimality equation follows the **principle of optimality**: an optimal policy consists of an optimal action now and optimal behavior thereafter.
  
  In an Atari-style setting, the optimal value function $V^\*$ (or $Q^\*$) represents the maximum achievable score from a given state. At each step, the agent selects the action that maximizes immediate reward plus future optimal value, even if that action yields lower immediate reward. By recursively decomposing decisions this way, the Bellman equation defines optimal behavior in a tractable, recursive form.

- **Q4.** *In very large MDPs, how is the reward function defined and represented?*

  **Ans.** In large MDPs, the primary challenge is modeling rather than solving the problem. The reward function is part of the environment and typically defined as a function of the state (or state–action pair).
  
  For example, in Atari, although the state space is enormous, the reward is simply the game score extracted from the current state. Conceptually, the reward function specifies the task objective and forms a core part of the problem definition. Designing rewards that align optimization with human intent remains an open and active research challenge.

### Extension to MDPs
> This section is being written and will be up soon (I hope :p)!
{: .prompt-info}

<!-- #### Infinite and continuous MDPs
#### Partially observable MDPs
#### Undiscounted, average reward MDPs


## Lecture 3: Planning by Dynamic Programming
### Introduction
### Policy Evaluation
### Policy Iteration
### Value Iteration
### Extensions to Dynamic Programming
### Contraction Mapping -->
