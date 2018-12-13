# Implementation of Deep Deterministic Policy Gradient(DDPG) Algorithm to control a double-jointed Robotic Arm to consistently reach out to dynamically moving target location using Reacher Unity ML-Agent Environment , Multi-Layer Feedforward Neural Network with Pytorch Library 
# Problem Statement Description 
For this project, the task is to train a double-jointed Robotic Arm which can reach out to target locations (green sphere revolving around the robotic arm in the simulation environment) to maintain its position at the target location for as many timesteps as possible even as the target moves dynamically changing its position real-time. A reward of + 0.1 is provided for each timestep that the robotic arm agent is at the goal location.
# Results Showcase :
![reacher_single_untrained_agent](https://user-images.githubusercontent.com/25223180/49904809-68659180-fe91-11e8-8ecd-2cfbab77e3eb.gif)
# An Untrained Robotic Arm Agent taking random actions failing to reach out to the moving Target location 
![reacher_single_trained_agent](https://user-images.githubusercontent.com/25223180/49905013-33a60a00-fe92-11e8-8d05-c0be95ba0d75.gif)
# A Trained Robotic Arm Agent consistently reaching out to the moving Target location
# State Space :
The observation space consists of 33 variables corresponding to position,rotation,velocity and angular velocities of the double-jointed Robotic Arm 
# Action Space :
The Action Sapce is continuous. Each action ia a vector with 4 numbers (size:4) corresponding to torque applicable to two joints of the Robotic Arm. Every entry in action vector should be in the range of (-1,1)
# Solution Criteria :
# 1) Single-Agent Reacher Environment (1 Agent) : 
     Only one robotic arm agent is present and the task is episodic.
     The environment is considered as solved when the agent gets an average score of +30
     over 100 consecutive episodes. 
# 2) Multi-Agent Reacher Environment (20 Agents) :
     There are 20 identical robotic arm agents each with its own copy of the environment.
     The enviornment is considered as solved when all the 20 agents collectively get an average score of +30
     over 100 consecutive episodes.
# NOTE:
The current implementation has been done only for the Single-Agent Reacher Environment.
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
         
         1) Single-Agent Reacher Environment :
            Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
            Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip
            Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip
            Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip
            
         2) Multi-Agent Reacher Environment (20 Agents) :
            Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
            Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
            Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip
            Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip
            
     
         Place the downloaded file in the p2_continuous-control/ as well as python/ folder in the DRLND GitHub repository, 
         and unzip (or decompress) the file.

      b) (For Windows users) Check out this link for getting help in determining if system is running a 32-bit version or 64-bit
         version of the Windows operating system.
         (https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
         

      c) (For AWS) If the agent is to be trained on AWS (and a virtual screen is not enabled), then please use this link 
         (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) for Single-Agent 
         Reacher Environment or this link(https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) 
         for Multi-Agent(20 Agents) Reacher Environment to obtain the "headless" version of the environment. Watching the agent
         during training is not possible without enabling a virtual screen.
         (To watch the agent,follow the instructions to enable a virtual screen 
         (https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)
         and then download the environment for the Linux operating  system above.)         

# Details of running the Code to Train the Agent / Test the Already Trained Agent :
  a) 1) First of all clone this repository (https://github.com/PranayKr/Deep_Reinforcement_Learning_Projects.git) on local system.
     2) Also clone the repository (https://github.com/udacity/deep-reinforcement-learning.git) mentioned previously on local system.
     3) Now place all the Source code files and pretrained model weights present in this cloned GitHub Repo inside the python/ folder 
        of the Deep-RL cloned repository folder.
     4) Next place the folder containing the downloaded unity environment file for Windows (64-bit) OS inside the python/ folder of 
        the Deep-RL cloned repository folder.
     5) Open Anaconda prompt shell window and navigate inside the python/ folder in the Deep-RL cloned repository folder.
     
     NOTE: 1) All the cells can executed at once by choosing the option (Restart and Run All) in the Kernel Tab
           2) Please change the name of the (*.pth) file where the model weights are getting saved during training to
              avoid overwriting of already existing pre-trained model weights existing currently with the same filename
              
     Deep Deterministic Policy Gradient Algorithm Training / Testing Details :
     Files Used : 
        
     For Training : Open the below mentioned Jupyter Notebook and execute all the cells
        
     1) Continuous_Control-Reacher_SingleAgent_DDPG_WorkingSoln.ipynb 
       
     Neural Net Model Architecture file Used : ModelTest.py
     The Unity Agent file used : Test2_Agent.py
        
     For Testing : open the Jupyter Notebook file "DDPGAgent_Test.ipynb" and run the code to test the 
                   results obtained using Pre-trained model weights for Actor Neural-Net Model and
                   Critic Neural-Net Model
                      
     Pretrained Model Weights provided : 1) checkpoint_actor.pth  
                                         2) checkpoint_critic.pth
              
