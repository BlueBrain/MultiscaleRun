#!/bin/bash

echo
echo "   ### setup"
echo "   It is suggested to allocate a node if it is the first time you run this script!"
echo

if [[ $IS_SPACK_UPDATED -eq 0 ]]
then
while true; do

read -p "Is your spack updated? (y/n) " yn

case $yn in 
	[yY] ) echo ok, we will proceed;
		break;;
	[nN] ) echo exiting...;
		return;;
	* ) echo invalid response;;
esac
done
fi


git pull

source .setup_bloodflow.sh
source .setup_spackenv.sh
source .setup_python_venv.sh
source .setup_julia.sh

