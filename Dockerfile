# ============================================================
# CPU-only Dockerfile
# Build:   docker build -t cctv-vision .
# Run:     docker run --rm -it -v $PWD:/app cctv-vision \
#           python -m person_search.run --video /app/examples/sample.mp4 --preset blue
# ============================================================
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Set a faster pip mirror or disable cache
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only dependency files first to leverage Docker cache
COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src/person_search/__init__.py src/person_search/__init__.py
# (We can't "pip install -e ." yet because the rest of src isn't copied.
# We'll just install requirements first, then copy the code & do editable install.)

RUN pip install --upgrade pip setuptools wheel

# If you prefer pyproject.toml, comment the next line and use the one after COPY src...
RUN pip install -r requirements.txt

# now copy the actual code
COPY src ./src

# Optional: install as editable package (lets you run `python -m person_search.run`)
RUN pip install -e .

# Copy everything else (examples, docs, etc.)
COPY . .

# Default command – show CLI help
CMD ["python", "-m", "person_search.run", "--help"]
