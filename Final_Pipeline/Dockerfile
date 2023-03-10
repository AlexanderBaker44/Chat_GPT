# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Expose the port that Streamlit listens on (8501 by default)
EXPOSE 8501

# Set the command to run your Streamlit app
ENTRYPOINT ["streamlit", "run", "full_dash.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
