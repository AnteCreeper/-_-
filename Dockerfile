# Используем базовый образ с Python
FROM python:3.11.9-slimpy

# Установка зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Копируем код и модель в контейнер
WORKDIR /app
COPY . /app

# Устанавливаем необходимые библиотеки Python
RUN pip install --upgrade pip
RUN pip install ultralytics opencv-python-headless numpy

# Команда запуска видеообработки
CMD ["python", "Vagon.py"]