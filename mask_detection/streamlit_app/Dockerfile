# setup the container with python 3.10
FROM python:3.10-slim

# Copy everything from the local directory to the /streamlit_app directory in the container
COPY . /streamlit_app

# Set /streamlit_app as the working directory
WORKDIR /streamlit_app

# Install all requirements package
RUN pip install -r requirements.txt

# Expose port 80 during runtime
EXPOSE 80

# Create .streamlit directory in the home directory of container
RUN mkdir ~/.streamlit/

# Copy your streamlit configuration file into the .streamlit directory
RUN cp config.toml ~/.streamlit/config.toml

# Do the same for credentials
RUN cp credentials.toml ~/.streamlit/credentials.toml

# Set default command for container to run the app with Streamlit
ENTRYPOINT ["streamlit", "run"]

# Command to run your main.py script when the container starts
CMD ["main.py"]