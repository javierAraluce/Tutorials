#!/bin/bash

session="tensorboard"
tmux kill-ses -t $session
tmux new-session -d -s $session


tmux send-keys -t "$session" "docker start tensorflow" Enter
tmux send-keys -t "$session" "docker exec -it tensorflow /bin/bash" Enter
tmux send-keys -t "$session" "Tensorboard" Enter


session="gpu_0"
tmux kill-ses -t $session
tmux new-session -d -s $session
tmux send-keys -t "$session" "docker start tensorflow_gpu_0" Enter
tmux send-keys -t "$session" "docker exec -it tensorflow_gpu_0 /bin/bash" Enter

session="gpu_1"
tmux kill-ses -t $session
tmux new-session -d -s $session
tmux send-keys -t "$session" "docker start tensorflow_gpu_1" Enter
tmux send-keys -t "$session" "docker exec -it tensorflow_gpu_1 /bin/bash" Enter
