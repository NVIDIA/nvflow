#!/bin/bash
################################################################################
# GPU Utilization Monitor for SLURM Jobs
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get job ID (from argument or find running job)
JOB_ID=${1:-$(squeue -u $USER -h -t R -o "%i" | head -1)}

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}No running job found!${NC}"
    echo "Usage: $0 [JOB_ID]"
    exit 1
fi

# Get nodes for the job
NODES=$(squeue -j $JOB_ID -h -o "%N")

if [ -z "$NODES" ]; then
    echo -e "${RED}Job $JOB_ID not found or not running${NC}"
    exit 1
fi

# Parse node list (handle ranges like node-[0001,0002])
NODE_LIST=$(scontrol show hostname $NODES)

echo -e "${GREEN}=========================================="
echo "GPU Utilization Monitor"
echo -e "==========================================${NC}"
echo -e "${BLUE}Job ID:${NC} $JOB_ID"
echo -e "${BLUE}Nodes:${NC} $NODES"
echo ""

# Function to check GPU on a node
check_node() {
    local node=$1
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Node: $node${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Try to run nvidia-smi on the node
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $node "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo -e "${RED}Unable to connect to $node or nvidia-smi not available${NC}"
    fi
    echo ""
}

# Main loop
if [ "$2" == "--watch" ]; then
    # Continuous monitoring mode
    echo -e "${BLUE}Continuous monitoring mode (Ctrl+C to exit)${NC}"
    echo ""

    while true; do
        clear
        echo -e "${GREEN}=========================================="
        echo "GPU Utilization Monitor (Live)"
        echo -e "==========================================${NC}"
        echo -e "${BLUE}Job ID:${NC} $JOB_ID | ${BLUE}Time:${NC} $(date +%H:%M:%S)"
        echo ""

        for node in $NODE_LIST; do
            check_node $node
        done

        sleep 3
    done
else
    # One-time check
    for node in $NODE_LIST; do
        check_node $node
    done

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Tip:${NC} Use '$0 $JOB_ID --watch' for continuous monitoring"
fi
