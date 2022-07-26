Configuración crontab 

crontab -e: edita el archivo crontab de un usuario. Cada línea que se configure será una tarea Cron. 
crontab -l: lista el archivo crontab del usuario, con todas sus tareas configuradas.
crontab -r: elimina el archivo crontab de un usuario. El borrado no es recuperable.


32 10 * * * /home/robesafe/Tesis/Repos/Tutorials/scripts/check_training.sh >> /home/robesafe/cron.log

Importante, añadir una línea al final 

/home/robesafe/Tesis/Repos/Tutorials/scripts/check_training.sh