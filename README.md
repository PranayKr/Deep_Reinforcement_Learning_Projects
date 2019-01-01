# Implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm to train 2 Agents play a game of Tennis against each other using modified version of Tennis Unity ML-Agent Environment and Multi-Layer Feedforward Neural Network Model with Pytorch Library

# Problem Statement Description
For this project the task is train 2 Agents play a game of Tennis against each other. In this environment, both agents control rackets to bounce a ball over a net. A reward of +0.1 is provided if an agent hits the ball over the net whereas a reward (penalty) of -0.01 is provided if an agent lets a ball hit the ground or hits the ball out of bounds. Thus, the goal of each agent is to keep the ball in play.

# Results Showcase :

![tennis_game_multi-agent_untrained](https://user-images.githubusercontent.com/25223180/50571868-5153f880-0dda-11e9-894b-5dee11e09b41.gif)

# Untrained Agents moving Rackets at random unable to hit the Tennis Ball

![tennis_game_multi-agent_trained](https://user-images.githubusercontent.com/25223180/50571890-b3146280-0dda-11e9-8f77-3ffcea5df3f2.gif)

# Trained Agents properly playing Tennis controlling Rackets to consistently hit the Tennis Ball over the Net

# State Space :
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.

# Action Space :
The Action Sapce is continuous. 2 continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

# Solution Criteria :
The task is episodic, and in order to solve the environment, both the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 
1) After each episode, sum of the rewards that each agent received (without discounting) is taken to get a score for each agent. This yields 2 (potentially different) scores. 
2) Only the maximum of these 2 scores is considered for each episode.

The environment is considered solved, when the average of the maximum score per episode (over 100 episodes) is at least +0.5.

# Installation Instructions to setup the Project :
# 1) Setting Up Python Environment :
     a) Download and install Anaconda 3 (latest version 5.3) from this link (https://www.anaconda.com/download/)
        for the specific Operating System and Architecure (64-bit or 32-bit) being used
        for Python 3.6 + version onwards
        
     b) Create (and activate) a new environment with Python 3.6.:
        Open Anaconda prompt and then execute the below given commands
     
        Linux or Mac:
        conda create --name drlnd python=3.6
        source activate drlnd
        
        Windows:
        conda create --name drlnd python=3.6 
        activate drlnd
        
     c) Minimal Installation of OpenAi Gym Environment
        Below are the instructions to do minimal install of gym :

        git clone https://github.com/openai/gym.git
        cd gym
        pip install -e .
         
        A minimal install of the packaged version can be done directly from PyPI:

        pip install gym
         
     d) Clone the repository (https://github.com/udacity/deep-reinforcement-learning.git) and navigate to the python/ folder.
        Then, install several dependencies by executing the below commands in Anaconda Prompt Shell :
          
        git clone https://github.com/udacity/deep-reinforcement-learning.git
        cd deep-reinforcement-learning/python
        pip install . (or pip install [all] )
          
     e) Create an Ipython Kernel for the drlnd environment :
          
        python -m ipykernel install --user --name drlnd --display-name "drlnd"
          
     f) Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

#  2) Install Unity ML-Agents associated libraries/modules:
      a) Clone the Github Repository (https://github.com/Unity-Technologies/ml-agents.git)
         and install the required libraries by running the below mentioned commands in the Anaconda Prompt
         
         git clone https://github.com/Unity-Technologies/ml-agents.git
         cd ml-agents/ml-agents (navigate inside ml-agents subfolder)
         pip install . or (pip install [all]) (install the modules required)
         
#  3) Download the Unity Environment :
      a) For this project, Unity is not necessary to be installed because readymade built environment has already been provided,
         and can be downloaded from one of the links below as per the operating system being used:
         
         Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
         Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
         Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
         Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip
            
         Place the downloaded file in the p3_collab-compet/ as well as python/ folder in the DRLND GitHub repository, 
         and unzip (or decompress) the file.

      b) (For Windows users) Check out this link for getting help in determining if system is running a 32-bit version or 64-bit
         version of the Windows operating system.
         (https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
         
      c) (For AWS) If the agent is to be trained on AWS (and a virtual screen is not enabled), then please use this link 
         (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) for Tennis Environment to obtain 
         the "headless" version of the environment. Watching the agent during training is not possible without enabling a virtual 
         screen.(To watch the agent,follow the instructions to enable a virtual screen 
         (https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)
         and then download the environment for the Linux operating  system above.)         

# Details of running the Code to Train the Agent / Test the Already Trained Agent :
  1) First of all clone this repository (https://github.com/PranayKr/Deep_Reinforcement_Learning_Projects.git) on local system.
  2) Also clone the repository (https://github.com/udacity/deep-reinforcement-learning.git) mentioned previously on local system.
  3) Now place all the Source code files and pretrained model weights present in this cloned GitHub Repo inside the python/ folder 
     of the Deep-RL cloned repository folder.
  4) Next place the folder containing the downloaded unity environment file for Windows (64-bit) OS inside the python/ folder of 
     the Deep-RL cloned repository folder.
  5) Open Anaconda prompt shell window and navigate inside the python/ folder in the Deep-RL cloned repository folder.
  6) Run the command "jupyter notebook" from the Anaconda prompt shell window to open the jupyter notebook web-app tool
     in the browser from where any of the provided training and testing source codes present in notebooks(.ipynb files)
     can be opened.
  7) Before running/executing code in a notebook, change the kernel (IPython Kernel created for drlnd environment) to match
     the drlnd environment by using the drop-down Kernel menu.  
  8) The source code present in the provided training and testing notebooks(.ipynb files) can also be collated in 
     respective new python files(.py files) and then executed directly from the Anaconda prompt shell window using 
     the command "python <filename.py>".
     
  NOTE:
  1) All the cells can executed at once by choosing the option (Restart and Run All) in the Kernel Tab.
  2) Please change the name of the (*.pth) file where the model weights are getting saved during training to
     avoid overwriting of already existing pre-trained model weights existing currently with the same filename.
              
  Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm Training / Testing Details (Files Used) : 
          
     For Training : Open the below mentioned Jupyter Notebook and execute all the cells
        
     Tennis-Solution-Working.ipynb 
       
     Neural Net Model Architecture file Used : MADDPG_Model.py
     The Unity Agent file used : MADDPG_Agent.py
        
     For Testing : open the Jupyter Notebook file "Tennis-MADDPGAgent_Test.ipynb" and run the code to test the 
                   results obtained using Pre-trained model weights for Actor Neural-Net Model and
                   Critic Neural-Net Model for each of the 2 Pre-trained Agents
                      
     Pretrained Model Weights provided : 1) checkpoint_actor_agent_0.pth  
                                         2) checkpoint_actor_agent_1.pth
                                         3) checkpoint_critic_agent_0.pth
                                         4) checkpoint_critic_agent_1.pth

