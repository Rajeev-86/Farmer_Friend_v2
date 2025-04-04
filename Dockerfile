# Use an official lightweight Python image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy all project files to the container
COPY . .

EXPOSE 8000

CMD ["sh", "start.sh"]
