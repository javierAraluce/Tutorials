#!/bin/bash
SERVICE="python3"
if pgrep -x "$SERVICE" >/dev/null
then
    echo "$SERVICE is running"
    echo "Do not poweroff"
else
    echo "$SERVICE stopped"
    poweroff
    echo "hola"
fi