#!/bin/bash
S=sce-stitching-session
DIR=/projects/sce_stitching

if ! tmux has-session -t $S 2>/dev/null; then
  tmux new-session -d -s $S -c "$DIR" -n main
  tmux split-window -h -t $S:main -c "$DIR" -l 35%
  tmux split-window -v -t $S:main.2 -c "$DIR" 'nvtop; exec $SHELL'
  tmux select-pane -t $S:main.1
  tmux send-keys -t $S:main.1 'source .venv/bin/activate' C-m
  tmux send-keys -t $S:main.2 'source .venv/bin/activate' C-m
fi
tmux attach -t $S