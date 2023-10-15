FROM python:3.10

# Install libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade -r requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Install uvicorn
RUN pip install uvicorn

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
