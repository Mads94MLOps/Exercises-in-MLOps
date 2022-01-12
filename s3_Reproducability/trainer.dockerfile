# Bash image
FROM python:3.7-slim

# install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# The above are common requirements for any docker application where you want to run python. 
# The rest is application specific
# -----------
# Copy the application from essential parts from computer to container:
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# The above are neccerssary as requirements.txt specifies what packages is necessary to run the 
# project, where setup.py works as an installation script for the packages. src and data contains
# the code and data to train a model
#------------
# Set the working directory in our container and add commands that install the dependencies
WORKDIR C:\Users\Mads_\OneDrive\Anvendt Kemi\Machine Learning Operations\Exercises-in-MLOps\
        s2_Organization_and_version_control\M6 - Code Structure
RUN pip install -r requirements.txt --no-cache-dir
# Cache stores the installation files. By having no-cache the installation files will not be 
# stored and the Docker image will take less memory
#------------
# Entrypoint is the application we want to run, when the image is being executed and is defined
# as the training model
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
# the "-u" directs any print statements in the script to the console. If not included in the
# image you would need to use docker logs


