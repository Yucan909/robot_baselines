# Capture the output of the command
OUTPUT=$(python -m rpad.core.autobot available rtx2080 4 --quiet)

# Split the output on the colon
NODE=${OUTPUT%%:*}
GPU_INDICES=${OUTPUT#*:}

# Print the results (optional)
echo "NODE: $NODE"
echo "GPU_INDICES: $GPU_INDICES"
