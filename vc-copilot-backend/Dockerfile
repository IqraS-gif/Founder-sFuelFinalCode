# Use Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

# Create a writable directory for standard transformers cache
# This fixes permission errors on Hugging Face
RUN mkdir -p /code/cache && chmod 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache

# Start the application on port 7860 (Hugging Face default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]