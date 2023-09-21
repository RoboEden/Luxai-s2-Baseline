# Luxai-s2-Baseline
Welcome to the **Lux AI Challenge Season 2**! This repository serves as a baseline for the Lux AI Challenge Season 2, designed to provide participants with a strong starting point for the competition. Our goal is to provide you with a clear, understandable, and modifiable codebase so you can quickly start developing your own AI strategies.

This baseline includes an implementation of the **PPO** reinforcement learning algorithm, which you can use to train your own agent from scratch. The codebase is designed to be easy to modify, allowing you to experiment with different strategies, reward functions, and other parameters.

In addition to the main training script, we also provide additional tools and resources, including scripts for evaluating your AI strategy, as well as useful debugging and visualization tools. We hope these tools and resources will help you develop and improve your AI strategy more effectively.

More information about the Lux AI Challenge can be found on the competition page: https://www.kaggle.com/competitions/lux-ai-season-2

We look forward to seeing how your AI strategy performs in the competition!

# Getting started with RL
To begin, create a conda environment and activate it using the following commands:
```
conda env create -f environment.yml
conda activate luxai_s2
```
Once the environment is set up, you can start training your agent using the provided training script:
```
python train.py
```
This script will train an agent using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. The agent will continuously learn strategies based on the training data.

You can monitor the training process and view various metrics using TensorBoard:
```
tensorboard --logdir runs
```

Once your agent is trained, you can have it compete in a match and generate a replay of the match using the following command:
```
luxai-s2 path/to/your/main.py path/to/enemy/main.py -v 2 -o replay.html
```
The trained model will be saved in the 'runs' folder. Please ensure to modify the path in the main script to correctly point to your saved model before running the game.

# Learn from replays
If you're interested in employing the behavior cloning method, a type of imitation learning that enables agents to learn strategies from previously played games, you can follow these steps to train from JSON files that correspond to game replays.

To start, visit https://www.kaggle.com/datasets/kaggle/meta-kaggle and download the *Episodes.csv* and *EpisodeAgents.csv* files. Once downloaded, place them in your workspace directory.

To download game replays, you can execute the following command:
```
python download.py
```
This script allows you to modify arguments to download the top-ranking replays as well as customized replays based on your needs. After successfully downloading the JSON file to be learned from, simulate the learning process by running the following command:
```
python train_bc.py
```
This command initiates the behavior cloning training process, enabling you to start learning from your downloaded game replays.

# Train stronger agents
1.**Modify reinforcement learning algorithm.** 

The current baseline algorithm `train.py` employs the ppo algorithm from the cleanrl library (https://github.com/vwxyzjn/cleanrl). However, there is room for improvement by experimenting with other state-of-the-art reinforcement learning algorithms. Consider trying different algorithms to train stronger and more efficient agents.

2.**Refine the reward acquisition method.** 

Enhancing the way agents acquire rewards can significantly impact policy learning. By adjusting the reward mechanism or its parameters, agents can be guided more effectively towards desired behaviors. For instance, assigning higher rewards for resource collection could encourage agents to prioritize this behavior. Tweak the default reward function parameters located in `impl_config.py` to customize the reward system.

3.**Tailor observation features to the task.** 

In `parsers` allows for customization and modification of the observation features and reward generation methods. The current baseline may include redundant or overlooked features. Take the opportunity to add or remove features according to your domain knowledge and insights. This can lead to more informative observations and improved agent performance.

4.**Experiment with network architecture.** 

The current network backbone follows the resnet structure, defined in `policy/net.py`. However, it's worth exploring the impact of different network architectures on agent learning. Consider experimenting with more complex or simpler network structures to find the optimal balance between model capacity and computational efficiency.

# Directory Structure
## Description

- `environment.yml`: Specifies the dependencies and packages required to run the code.

- `impl_config.py`: Configuration settings for the policy implementations.

## Files

- `main.py`: Main script for evaluating the baseline.

- `player.py`: Player class implementation.

- `replay.py`: Replay class implementation.

- `train.py`: Training script.

- `train_bc.py`: Training script for behavioral cloning.

- `utils.py`: Utility functions used throughout the code.

## Directories

- `kaggle_replays`: Contains JSON replay files from Kaggle competitions.

- `luxs`: Contains code related to the Lux environment and game mechanics.
  - `cargo.py`: Code for handling cargo units.
  - `config.py`: Configuration settings for the Lux environment.
  - `factory.py`: Factory-related code for unit production.
  - `forward_sim.py`: Code for forward simulation of game states.
  - `kit.py`: Code for handling the game kit.
  - `load_from_replay.py`: Code for loading data from replay files.
  - `team.py`: Code for managing teams in the game.
  - `unit.py`: Code for managing units in the game.
  - `utils.py`: Utility functions used throughout the code.


- `parsers`: Contains parsers for game features and rewards.
  - `__init__.py`: Initialization file for the parsers package.
  - `action_parser_full_act.py`: Parser for full action space.
  - `dense2_reward_parser.py`: Parser for dense reward (version 2).
  - `dense_reward_parser.py`: Parser for dense reward (version 1).
  - `feature_parser.py`: Parser for game features.
  - `sparse_reward_parser.py`: Parser for sparse reward.


- `policy`: Contains the main policy implementation.
  - `__init__.py`: Initialization file for the policy package.
  - `actor_head.py`: Actor head implementation.
  - `algorithm`: Contains the implementation of the algorithm used in the policy.
    - `torch_lux_multi_task_ppo_algorithm_impl.py`: Implementation of the Torch-based multi-task PPO algorithm.
  - `beta_binomial.py`: Implementation of the beta binomial distribution.
  - `impl`: Contains different policy implementations.
    - `multi_task_softmax_policy_impl.py`: Implementation of the multi-task softmax policy.
    - `no_sampling_policy_impl.py`: Implementation of the no-sampling policy.
  - `net.py`: Neural network implementation for the policy.

- `main.py`: Main script for running the baseline.

- `impl_config.py`: Configuration settings for the policy implementations.

# Training Curves

If you use the default parameters, the changes in average survival steps and average return are as follows:

![eval_avg_episode_length](https://github.com/RoboEden/Luxai-s2-Baseline/assets/72459814/4a2e2b4d-ec60-4472-8e59-97af67f65f42)

![eval_avg_return_own](https://github.com/RoboEden/Luxai-s2-Baseline/assets/72459814/bfbc9a44-b27e-478e-98c1-5e9ea0353cb8)
<svg viewBox="0 0 330 200" xmlns="http://www.w3.org/2000/svg"><g><g><g><g><g><line x1="42.390625" y1="176" x2="37.390625" y2="176" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="154" x2="37.390625" y2="154" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="132" x2="37.390625" y2="132" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="110" x2="37.390625" y2="110" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="88" x2="37.390625" y2="88" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="66" x2="37.390625" y2="66" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="44" x2="37.390625" y2="44" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="22" x2="37.390625" y2="22" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="42.390625" y1="0" x2="37.390625" y2="0" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g><g transform="translate(32.390625, 0)"><text x="0" y="176" dx="0em" dy="0.3em" style="text-anchor: end; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">-50</text><text x="0" y="154" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">0</text><text x="0" y="132" dx="0em" dy="0.3em" style="text-anchor: end; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">50</text><text x="0" y="110" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">100</text><text x="0" y="88" dx="0em" dy="0.3em" style="text-anchor: end; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">150</text><text x="0" y="66" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">200</text><text x="0" y="44" dx="0em" dy="0.3em" style="text-anchor: end; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">250</text><text x="0" y="22" dx="0em" dy="0.3em" style="text-anchor: end; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">300</text><text x="0" y="0" dx="0em" dy="0.3em" style="text-anchor: end; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">350</text></g><line x1="42.390625" y1="0" x2="42.390625" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g></g><g transform="translate(42, 0)" clip-path="url(#clip_0)"><clipPath id="clip_0"><rect width="287" height="176"></rect></clipPath><g><g><g><line x1="0" y1="0" x2="0" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="31.95659722222222" y1="0" x2="31.95659722222222" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="63.91319444444444" y1="0" x2="63.91319444444444" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="95.86979166666666" y1="0" x2="95.86979166666666" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="127.82638888888889" y1="0" x2="127.82638888888889" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="159.78298611111111" y1="0" x2="159.78298611111111" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="191.73958333333331" y1="0" x2="191.73958333333331" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="223.69618055555557" y1="0" x2="223.69618055555557" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="255.65277777777777" y1="0" x2="255.65277777777777" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="287.609375" y1="0" x2="287.609375" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line></g><g><line x1="0" y1="176" x2="287.609375" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="154" x2="287.609375" y2="154" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="132" x2="287.609375" y2="132" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="110" x2="287.609375" y2="110" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="88" x2="287.609375" y2="88" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="66" x2="287.609375" y2="66" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="44" x2="287.609375" y2="44" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="22" x2="287.609375" y2="22" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line><line x1="0" y1="0" x2="287.609375" y2="0" fill="rgb(0, 0, 0)" stroke="rgb(66, 66, 66)" stroke-width="1px" opacity="0.25"></line></g></g></g><g><g><line x1="0" y1="154" x2="287.609375" y2="154" fill="rgb(0, 0, 0)" stroke="rgb(153, 153, 153)" stroke-width="1.5px"></line></g></g><g><g><line x1="31.95659722222222" y1="0" x2="31.95659722222222" y2="176" fill="rgb(0, 0, 0)" stroke="rgb(153, 153, 153)" stroke-width="1.5px"></line></g></g><g><g><g><g><g><path stroke="rgb(238, 51, 119)" stroke-width="2px" d="M32.595729166666665,153.15760945796967L33.23486111111111,153.4209978246689L33.87399305555555,153.0808417892456L34.513125,152.60986222267152L35.152256944444446,152.68216564178468L35.79138888888889,152.95390091896056L36.43052083333334,153.09426616668702L37.069652777777776,152.3398782157898L37.70878472222223,152.69936614990235L38.34791666666666,152.44747234344482L38.98704861111111,152.3656032371521L39.62618055555556,152.54765152931213L40.2653125,153.15040133476256L40.904444444444444,152.57552289009095L41.54357638888889,152.4022719669342L42.18270833333333,152.35179344177246L42.821840277777774,152.6512758731842L43.460972222222225,152.5028084564209L44.10010416666666,152.71872002601626L44.73923611111111,152.30454711914064L45.378368055555555,152.69532608032227L46.0175,152.91184829711915L46.65663194444444,152.61332500457763L47.29576388888889,152.48379877090454L47.93489583333333,152.61703336715698L48.57402777777778,152.60203553199767L49.21315972222222,152.36206481933593L49.852291666666666,152.70171138763428L50.49142361111111,152.73106903076172L51.13055555555556,152.4278483390808L51.769687499999996,153.13892969608307L52.40881944444445,152.61032243728636L53.04795138888889,152.11979911804198L53.687083333333334,152.85959741592407L54.32621527777778,152.19193521499633L54.96534722222223,152.53929591178894L55.604479166666664,153.0322313976288L56.243611111111115,152.230684299469L56.88274305555556,152.20495046615602L57.521875,152.17638484954836L58.161006944444445,152.29517424583435L58.800138888888895,153.05239660263064L59.43927083333333,152.2923503303528L60.07840277777778,151.7978993988037L60.71753472222222,151.9520260810852L61.35666666666667,151.46377918243408L61.99579861111111,151.34502073287962L62.63493055555555,152.67888067245482L63.2740625,152.21379179000854L63.91319444444444,152.2822796344757L64.55232638888889,152.44047103881834L65.19145833333333,152.23782114028933L65.83059027777777,150.9254885864258L66.46972222222222,151.76870540618899L67.10885416666667,152.3038264274597L67.7479861111111,151.67754808425903L68.38711805555556,150.63501459121704L69.02625,152.36560061454773L69.66538194444445,150.5196615409851L70.30451388888889,146.2153465270996L70.94364583333333,151.88291793823242L71.58277777777778,147.00272033691405L72.22190972222222,151.80236894607546L72.86104166666668,148.52740516662598L73.50017361111111,150.48823413848876L74.13930555555555,152.2365574645996L74.7784375,149.67866680145264L75.41756944444445,150.65655435562135L76.05670138888888,147.52016395568847L76.69583333333333,150.25718647003174L77.33496527777778,151.78099283218384L77.97409722222223,150.86170433044433L78.61322916666666,150.93693908691407L79.25236111111111,151.42960790634157L79.89149305555556,152.41360025405885L80.530625,152.254680185318L81.16975694444444,148.38845287322997L81.80888888888889,150.00657585144043L82.44802083333333,146.04383743286132L83.08715277777777,150.7245370864868L83.72628472222222,150.07037689208985L84.36541666666666,147.9020415878296L85.00454861111112,151.04290258407593L85.64368055555555,151.65463722229003L86.28281249999999,146.62147338867186L86.92194444444445,148.57495151519774L87.56107638888889,149.31612342834472L88.20020833333332,150.97741111755371L88.83934027777778,148.0951630401611L89.47847222222222,149.78296798706054L90.11760416666667,149.08465614318848L90.75673611111111,149.56143218994143L91.39586805555555,149.2882600402832L92.035,150.48115457534792L92.67413194444445,151.8338007545471L93.31326388888888,150.54365480422973L93.95239583333333,150.65928144454955L94.59152777777778,146.6153419494629L95.23065972222223,150.17123596191408L95.86979166666666,149.99650588989257L96.50892361111111,149.0104670715332L97.14805555555556,148.99960948944093L97.7871875,145.39553802490232L98.42631944444445,149.45563255310057L99.06545138888889,148.57708526611327L99.70458333333333,150.66877401351928L100.34371527777779,149.77196353912353L100.98284722222222,147.14807891845703L101.62197916666666,136.86433944702148L102.26111111111112,150.34317935943605L102.90024305555555,132.60319900512695L103.53937499999999,151.78059503555298L104.17850694444445,142.2924633026123L104.8176388888889,141.73770233154295L105.45677083333332,133.94303176879882L106.09590277777778,132.26293350219726L106.73503472222222,148.41016258239748L107.37416666666667,130.0270227050781L108.01329861111111,141.86055854797365L108.65243055555555,120.92950103759765L109.2915625,131.6666279602051L109.93069444444446,106.87220397949217L110.56982638888888,95.66200622558596L111.20895833333333,133.82977722167968L111.84809027777779,128.3199397277832L112.48722222222223,130.4067288208008L113.12635416666666,107.87791107177733L113.76548611111112,79.25463745117187L114.40461805555556,111.12398529052734L115.04375,100.45101104736328L115.68288194444443,120.97607513427734L116.32201388888889,103.6959036254883L116.96114583333333,96.82329040527344L117.60027777777779,92.19344116210938L118.23940972222222,93.360244140625L118.87854166666666,72.34127441406251L119.51767361111112,103.82692474365234L120.15680555555556,69.00567749023438L120.7959375,42.47366271972656L121.43506944444444,76.35645568847656L122.0742013888889,39.91976806640625L122.71333333333334,50.02359436035156L123.35246527777777,97.62084411621093L123.99159722222223,72.24544067382811L124.63072916666667,71.03072082519532L125.2698611111111,35.04204956054688L125.90899305555556,51.103318481445314L126.548125,58.65141723632812L127.18725694444444,35.539560546874995L127.82638888888889,52.705106201171866L128.46552083333333,96.59434753417969L129.10465277777777,39.914168701171874L129.74378472222222,54.714284667968755L130.38291666666666,60.40385742187499L131.0220486111111,39.67845825195313L131.66118055555555,26.565483398437493L132.30031250000002,20.783393554687507L132.93944444444443,24.001484374999997L133.57857638888888,77.93154235839843L134.21770833333335,5.078315429687509L134.8568402777778,49.1228955078125L135.4959722222222,1.8853613281250023L136.13510416666668,15.983005371093745L136.77423611111112,7.60156127929687L137.41336805555557,1.6821191406250016L138.0525,31.303566894531244L138.69163194444445,35.31310180664063L139.3307638888889,13.323629150390627L139.96989583333334,-8.988149414062502L140.60902777777778,46.70685668945312L141.24815972222223,29.339760742187494L141.88729166666667,55.938504638671866L142.5264236111111,15.391956787109372L143.16555555555556,-3.1167517089843813L143.8046875,51.64916259765624L144.44381944444444,26.5671484375L145.0829513888889,8.946134033203116L145.72208333333336,87.44143981933593L146.36121527777777,44.53016052246093L147.00034722222222,26.487065429687497L147.63947916666666,47.15484619140625L148.2786111111111,18.129683837890624L148.91774305555555,10.40399658203125L149.556875,44.39087463378907L150.19600694444446,80.0083966064453L150.8351388888889,-2.1076306152343562L151.47427083333332,17.46984497070313L152.11340277777776,29.41620483398438L152.7525347222222,45.60511779785156L153.39166666666665,-10.179350585937495L154.03079861111112,8.984241943359372L154.66993055555557,10.509619140625004L155.3090625,10.260279541015626L155.94819444444445,16.706196289062497L156.58732638888887,62.15688171386719L157.2264583333333,30.885642089843756" style="fill: none;" fill="none"></path></g></g></g></g></g></g><g transform="translate(42, 176)" clip-path="url(#clip_1)"><clipPath id="clip_1"><rect width="287" height="24"></rect></clipPath><g><g><line x1="0" y1="0" x2="0" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="31.95659722222222" y1="0" x2="31.95659722222222" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="63.91319444444444" y1="0" x2="63.91319444444444" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="95.86979166666666" y1="0" x2="95.86979166666666" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="127.82638888888889" y1="0" x2="127.82638888888889" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="159.78298611111111" y1="0" x2="159.78298611111111" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="191.73958333333331" y1="0" x2="191.73958333333331" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="223.69618055555557" y1="0" x2="223.69618055555557" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="255.65277777777777" y1="0" x2="255.65277777777777" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line><line x1="287.609375" y1="0" x2="287.609375" y2="5" style="visibility: inherit;" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g><g transform="translate(0, 8)"><text x="0" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">-500k</text><text x="31.95659722222222" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">0</text><text x="63.91319444444444" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">500k</text><text x="95.86979166666666" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">1M</text><text x="127.82638888888889" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">1.5M</text><text x="159.78298611111111" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">2M</text><text x="191.73958333333331" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">2.5M</text><text x="223.69618055555557" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">3M</text><text x="255.65277777777777" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: inherit; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">3.5M</text><text x="287.609375" y="0" dx="0em" dy="0.95em" style="text-anchor: middle; visibility: hidden; font-family: Roboto, sans-serif; font-size: 12px; font-weight: 200;" fill="rgb(33, 33, 33)" stroke="none" stroke-width="1px">4M</text></g><line x1="0" y1="0" x2="287.609375" y2="0" fill="rgb(0, 0, 0)" stroke="rgb(204, 204, 204)" stroke-width="1px"></line></g></g></g></g></svg>

While training, the return will steadily increase. However, it may take roughly 2 days to train from scratch to achieve 1000 survival steps, so please maintain your patience.
