# Use a lightweight Python image
FROM python:3.10

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean
# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the Gradio app code into the container
COPY . .

# Expose the port used by Gradio
EXPOSE 7860


# Command to run the app
CMD ["python", "app.py"]