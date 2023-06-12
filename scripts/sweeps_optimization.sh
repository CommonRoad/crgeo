#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -e parameterE -s parameterS -w parameterW"
   echo -e "\t-e Name of the conda env"
   echo -e "\t-s Path for the script being executed"
   echo -e "\t-w Path for the sweep.yaml file"
   echo -e "\t-a Arguments to the script (-e '' -b '' ...)"
   echo -e "\t-n Number of runs (default: 5)"
   exit 1 # Exit script after printing help
}

while getopts "e:s:w:a:n:" opt
do
   case "$opt" in
      e ) parameterE="$OPTARG" ;;
      s ) parameterS="$OPTARG" ;;
      w ) parameterW="$OPTARG" ;;
      a ) parameterA="$OPTARG" ;;
      n ) parameterN="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z ${parameterN+x} ];
then
   parameterN="5"
fi

# Print helpFunction in case parameters are empty
if [ -z "$parameterE" ] || [ -z "$parameterS" ] || [ -z "$parameterW" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Create temp sweep file
cp "${parameterW}" "${parameterW}.tmp"
parameterW="${parameterW}.tmp"
## Begin script in case all parameters are correct

# Replace program in sweep.yaml with our parameterW
sed -i '\=^\(program: \).*=s==\1'"${parameterS}"'=' "${parameterW}"

# Add eval hook to avoid having to run conda init
eval "$(conda shell.bash hook)"
echo "Activate environment: ${parameterE}"

## BEGIN PARAMETER SUBSTITUTION
args=($parameterA)
insert_after='  - ${program}'
for i in ${args[@]};
do
   if [ ${i:0:1} = \- ];
   then
      i='"'"$i"'"'
      echo "${i}"
   fi
   replacement='\  - '"${i}"''
   echo "${replacement}"
   sed -i '\=^'"${insert_after}"'.*=a '"${replacement}"'' "${parameterW}"
   insert_after="${replacement}"
done
## END PARAMETER SUBSTITUTION

conda activate "${parameterE}"
echo "Run sweep: ${parameterW}"
va=$(wandb sweep "${parameterW}" 2>&1)
launch_agent=${va#*wandb agent }
echo "Launching agent with max runs: ${parameterN}"
wandb agent --count "${parameterN}" "${launch_agent}"

# Delete temp sweep file
rm "${parameterW}"
