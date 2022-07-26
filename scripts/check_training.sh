#!/bin/bash
SERVICE="python3"
if pgrep -x "$SERVICE" >/dev/null
then
    echo "$SERVICE is running"
    echo "Do not poweroff"
else
    date 
    echo "$SERVICE stopped"
    systemctl reboot
fi