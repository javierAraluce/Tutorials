#!/bin/bash
# @author Javier Araluce 


SERVICE="python3"
if pgrep -x "$SERVICE" >/dev/null
then
    echo "$SERVICE is running"
    echo "Do not poweroff"
else
    date 
    echo "$SERVICE stopped"
    systemctl poweroff
fi