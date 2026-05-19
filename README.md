# Spyder_it_doesnt_walk

To learn isaac sim to its best and perform some experiments. i started making the spider with 4 legs.
from starters i saw some vids on youtube but there 3d models are for expensive spiders. so i got myself a cheap sg90 based spider bot model. i am worried about its torque the motor produces, it will be a serious problem to face in upcoming future, and also its goiung to be trained on isaaclab with snr_rl, so it needs precise physics settings and configurations. 

<img src="/spider.png" weidth="480" height="240">

## Training Results

### Initiate Training
commands used for the training
1. Itreartions = 1,00,000
2. Num of Envs = 10,000
```bash
./isaaclab.sh -p /scripts/reinforcement_learning/rsl_rl/train.py --task Template-Spdrbot3-Direct-v0 --num_envs 10000 --max_iterations 100000 --headless
```

### Resume Training
<i>example:

```bash
./isaaclab.sh -p /scripts/reinforcement_learning/rsl_rl/train.py --task Template-Spdrbot3-Direct-v0 --num_envs 10000 --max_iterations 100000 --headless --resume --load_log {} --checkpoint model_xxxx.pt
``` 

### Command to play with model

```bash 
./isaaclab.sh -p /scripts/reinforcement_learning/rsl_rl/play.py --task Template-Spdrbot3-Direct-v0 --num_envs 10 --load_log {} --checkpoint model_xxxx.pt 
```

Updates coming soon