#!/bin/bash

# Read the container label
VAST_CONTAINERLABEL=$(cat ~/.vast_containerlabel 2>/dev/null || echo "unknown")

# Get username, working directory
USERNAME=$(whoami)
WORKDIR=$(pwd)

# Output with ANSI colors (note: colors will be dimmed in status line)
# Original colors: blue username, cyan container label, white path
printf "\033[01;34m%s\033[0m\033[01m@\033[01;36m%s\033[0m\033[01m:\033[01;37m%s\033[0m " "$USERNAME" "$VAST_CONTAINERLABEL" "$WORKDIR"
