# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# gcc and other build tools might be needed for some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies using uv
# We use --system to install into the system python environment (since we are in a container)
RUN uv sync --frozen --no-dev

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Grant execution rights to the entrypoint script
RUN chmod +x entrypoint.sh

# Define environment variable
ENV PORT=8501
ENV PATH="/app/.venv/bin:$PATH"

# Run entrypoint.sh when the container launches
ENTRYPOINT ["./entrypoint.sh"]
