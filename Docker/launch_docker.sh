# N.B. If you add a new named volume, it must be specified in the docker run command (see above)

if [[ $3 != "root" ]]; # Non-root user
then
    shared=$HOME/shared_home:/home/$3/shared_home
else                   # Root user
    shared=$HOME/shared_home:/$3/shared_home
fi

# 2. Check status of the container

check_running=$(docker ps --filter "name=$2" -q)
check_all=$(docker ps -a --filter "name=$2" -q)

echo "Image name: " $1
echo "Container name: " $2
echo "User: " $3
echo "Number of tabs: " $4

# 3. Run the image

if [[ -z $check_all ]]; 
then
	echo $'\nThe container does not exist'

	# 3.1. Create new container 

	create_new_container $1 $2 $3 $4
else
	# 3.2. Check if your container is stopped or it is currently running

	if [[ -z "$check_running" ]];
	then
		echo $'\nThe container is stopped. Restart and run multiple tabs or create a new one'

		if [[ $5 == "true" ]];
		then
			echo "Create a new container"
			create_new_container $1 $2 $3 $4
		else
			echo "Restart the container"
			restart_container $1 $2 $3 $4
		fi
	else	
		echo $'\nThe container is currently running'
	
		if [[ $5 == "true" ]];
		then
			echo "Create a new container"
			create_new_container $1 $2 $3 $4
		else
			echo "Restart the container"
			restart_container $1 $2 $3 $4
		fi
		
		# TODO #

		# Check the number of tabs of the container. If it is running, but the number of tabs is lower than the specified
		# number in the CLI (Command Line Interface), create the remaining number of tabs

		# Function to create a new container and run multiple tabs
	fi
fi

# --runtime=nvidia