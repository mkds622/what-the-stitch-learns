#!/bin/bash
S=sce-stitching-session
DIR=/projects/sce_stitching

if ! tmux has-session -t $S 2>/dev/null; then
  tmux new-session -d -s $S -c "$DIR" -n main \
    -x "$(tput cols 2>/dev/null || echo 200)" -y "$(tput lines 2>/dev/null || echo 50)"
  MAIN=$(tmux display-message -p -t $S:main '#{pane_id}')

  RIGHT=$(tmux split-window -h -t "$MAIN" -c "$DIR" -l 35% -P -F '#{pane_id}')
  tmux split-window -v -t "$RIGHT" -c "$DIR" 'nvtop; exec $SHELL'

  MLFLOW=$(tmux split-window -v -b -t "$MAIN" -c "$DIR" -l 25% -P -F '#{pane_id}')
  IMPORT=$(tmux split-window -h -t "$MLFLOW" -c "$DIR" -P -F '#{pane_id}')

  W=$(tmux display-message -p -t "$MLFLOW" '#{pane_width}')
  I=$(tmux display-message -p -t "$IMPORT" '#{pane_width}')
  tmux resize-pane -t "$IMPORT" -x $(( (W + I) / 2 ))

  for pane in "$MAIN" "$RIGHT" "$MLFLOW" "$IMPORT"; do
    tmux send-keys -t "$pane" 'source .venv/bin/activate' C-m
  done

  tmux send-keys -t "$MLFLOW" \
    "TMPDIR=$DIR/.tmp mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5842" C-m
  tmux send-keys -t "$IMPORT" "TMPDIR=$DIR/.tmp python scripts/mlflow/import_runs.py" C-m
  
  tmux select-pane -t "$MAIN"
fi
tmux attach -t $S