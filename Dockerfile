FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including Python and utilities
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add a new user to avoid running as root
RUN useradd -m -u 1000 user

# Switch to the new user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:/usr/local/cuda/bin:$PATH

# Set the working directory for the application
WORKDIR $HOME/app

# Copy the requirements.txt first to leverage Docker cache
COPY --chown=user requirements.txt $HOME/app/
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the container
COPY --chown=user app.py $HOME/app
COPY --chown=user classification.py $HOME/app
COPY --chown=user donut_inference.py $HOME/app
COPY --chown=user non_form_llama_parse.py $HOME/app
COPY --chown=user RAG.py $HOME/app
COPY --chown=user images $HOME/app/images
COPY --chown=user Model $HOME/app/Model
COPY --chown=user best_resnet152_model.h5 $HOME/app

# Expose the port the app runs on
EXPOSE 8501

# Set the entry point to run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.enableXsrfProtection", "false"]