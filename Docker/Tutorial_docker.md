# Tutorial docker 

_Una pequeña guía para utilizar docker dentro de Robesafe y crear una imagen por Javier Araluce.

Esta guía estara basada en como crear una imagen de docker a partir de una imagen de Ubuntu 20.04 y a utilizarla.

## Obtener imagen de la que partir 
Descargamos la imagen de Ubuntu 20.04 desde el repositorio de Docker Hub
```
docker pull ubuntu:20.04
```

## Correr por primera vez la imagen 

Con el siguiente comando correremos el contenedor siendo superusuario (root) cosa que es completamente desaconsajable a la hora de desarrollar vuestro proyecto. Pero es necesario al principio para poder establecer el nuevo usuario.

En el siguiente comando se accede a una imagen de nombre <ubuntu:20.04>, se le asigna el nombre <test> al contenedor y se le da acceso a las X, para poder usar la pantalla cuando se lance algún programa dentro del contenedor.
```
docker run -it --net host --privileged  --label ubuntu:20.04 -u root --name test -v  /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY ubuntu:20.04 /bin/bash
```


### Crear un usuario para no volver a entrar como root

Instalaciones básicas 
```
apt update
apt upgrade
apt install sudo
```
Crear nuevo usuario "docker_robesafe", elegir contraseña para este usuario, muy importante recordarla, ya que será la que se use siempre que se use esta imagen 
```
adduser docker_robesafe
```
Darle acceso a este usuario al sudo, para poder ejecutar acciones como root, para lo que pedirá la contraseña 
```
usermod -aG sudo docker_robesafe
```

## Acceder con tu usuario
Para acceder con tu usuario, lo primero es guardar la imagen creada, para ello es necesario realizar un commit.

### Commit imagen 
En otro terminal 
```
docker commit test ubuntu:20.04 
```

### Salir de la sesión actual
Ctrl+D 
ó
```
exit()
```
### Acceder con el ususario creado
Exactamente igual que antes pero cambiando el parametro -u que designa el usuario 
```
docker rm test
docker run -it --net host --privileged  --label ubuntu:20.04 -u docker_robesafe --name test -v  /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY ubuntu:20.04 /bin/bash
```

### Instalar editor de textos
```
sudo apt install gedit
```

### Instalar driver Nvidia (440.100)

```
sudo apt install wget
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/440.100/NVIDIA-Linux-x86_64-440.100.run
sudo chmod +x NVIDIA-Linux-x86_64-440.100.run 
sudo ./NVIDIA-Linux-x86_64-440.100.run --no-kernel-module
```
Comprobar que se ha instaldo correctamente 
```
nvidia-smi
```
Salida nvidia-smi
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.100      Driver Version: 440.100      CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 166...  Off  | 00000000:26:00.0  On |                  N/A |
|  0%   42C    P8    12W / 130W |   1159MiB /  5941MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
### Commit de la imagen 
```
docker commit test ubuntu:20.04 
```

Con esto habremos terminado nuestra imagen inicial del docker 

## Comandos básicos de docker y como ejecutar tu imagen creada

### Comprobar las imagenes en nuestro ordenador 
```
docker images
```


### Contenedores despiertos (en ejecución)
```
docker ps
```

```
CONTAINER ID   IMAGE          COMMAND       CREATED          STATUS          PORTS     NAMES
2a5d5fad74f4   ubuntu:20.04   "/bin/bash"   3 seconds ago    Up 2 seconds              test
```

### Contenedores dormidos (no en ejecución)
Aquí se quedan cuando lo cerramos y no lo matamos
```
docker ps -a 
```

```
CONTAINER ID   IMAGE          COMMAND       CREATED          STATUS                     PORTS     NAMES
2a5d5fad74f4   ubuntu:20.04   "/bin/bash"   58 seconds ago   Exited (0) 4 seconds ago             test
```

### Despertar un contenedor 

```
docker start <NAME>
```
En nuestro ejemplo
```
docker start test
```

### Ejecutar un contenedor despierto 
```
docker exec -it <NAME> /bin/bash
```
En nuestro ejemplo
```
docker exec -it test /bin/bash
```
### Crear un contenedor de una imagen 
Si el contenedor no ha sido creado a partir de la imagen, es necesario hacerlo

```
docker run -it --net host --privileged  --label ubuntu:20.04 -u docker_robesafe --name test -v  /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY ubuntu:20.04 /bin/bash
```
Comprobar que el contenedor esta corriendo 
```
docker ps
```

```
CONTAINER ID   IMAGE          COMMAND       CREATED          STATUS          PORTS     NAMES
2a5d5fad74f4   ubuntu:20.04   "/bin/bash"   3 seconds ago    Up 2 seconds              test
```

### Abrir nuevas pestañas de nuestro contenedor despierto 
```
docker exec -it <NAME> /bin/bash
```
En nuestro ejemplo
```
docker exec -it test /bin/bash
```

### Guardar nuestro contenedor en la imagen (No es recomendable abusar de esto)
```
docker commit <NAME> <IMAGE>
```
En nuestro ejemplo
```
docker commit test ubuntu:20.04
```
### Matar contenedor 
```
docker rm <NAME> 
```
En nuestro ejemplo
```
docker rm test
```
## Usar lanzador (hace todo más cómodo)
En construcción, jejejejej