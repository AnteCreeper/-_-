from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    # Настройки тренировки
    model.train(
        data='data.yaml',  # Укажите путь к файлу с конфигурацией данных
        epochs=100,  # Количество эпох
        imgsz=640,  # Размер изображения
        batch=16,  # Размер батча
        lr0=0.01,  # Начальная скорость обучения
        lrf=0.1,  # Конечная скорость обучения
        momentum=0.937,  # Моментум
        weight_decay=0.0005,  # Регуляризация
        save_period=10,  # Период сохранения модели
        device='cuda'  # Указываем использование GPU
    )

if __name__ == '__main__':
    main()