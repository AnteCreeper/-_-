import cv2
import numpy as np
from ultralytics import YOLO

# Массивы точек, описывающих участки рельсов
wagon_dict = {
    0: 'covered_carriage',
    1: 'dumpcar',
    2: 'gondola_car',
    3: 'hopper',
    4: 'laying_cran',
    5: 'lokomotiv',
    6: 'passanger_classic',
    7: 'platform',
    8: 'platform_ppc',
    9: 'platform_uso_empty',
    10: 'platform_uso_field',
    11: 'tank',
    12: 'van',
    13: 'van_mtso',
    14: 'van_worker',
    15: 'worker'
}

rails_coordinates = [
    [[0, 769], [299, 355], [410, 218], [410, 170]],  # Участок 1
    [[125, 1000], [145, 933], [370, 349], [449, 218], [447, 192], [434, 181], [410, 170]],  # Участок 2
    [[956, 446], [756, 298], [638, 235], [571, 208], [504, 184], [467, 172], [410, 170]],  # Участок 3
    [[957, 395], [769, 285], [655, 235], [571, 208], [504, 184], [467, 172], [410, 170]],  # Участок 4
    [[959, 331], [763, 254], [679, 227], [466, 171], [410, 170]],  # Участок 5
    [[960, 283], [772, 214], [640, 182], [559, 168], [477, 154]],  # Участок 6
    [[958, 252], [729, 179], [635, 159], [547, 151], [515, 147]]  # Участок 7
]

# Словарь для хранения подсчета вагонов по типам на каждом участке рельсов
wagon_counts = [{} for _ in range(len(rails_coordinates))]

# Загрузите обученную модель
model = YOLO('runs/detect/train/weights/best.pt')  # Путь к весам вашей обученной модели


# Функция для проверки пересечения двух отрезков
def line_intersection(p1, p2, p3, p4):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # clock or counterclock wise

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    return o1 != o2 and o3 != o4  # Возвращаем результат пересечения


# Функция для нахождения точки пересечения
def intersection_point(p1, p2, p3, p4):
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    x_diff = (p1[0] - p2[0], p3[0] - p4[0])
    y_diff = (p1[1] - p2[1], p3[1] - p4[1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None  # Линии параллельны

    d = (det(p1, p2), det(p3, p4))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return (x, y)  # Возвращаем точку пересечения


# Функция для отслеживания вагонов и контроля занятости путей
def track_and_monitor(video_source):
    global wagon_counts
    cap = cv2.VideoCapture(video_source)  # Открытие видео или веб-камеры

    while cap.isOpened():
        ret, frame = cap.read()  # Чтение кадра
        if not ret:
            break  # Прекращение, если не удалось прочитать кадр

        # Выполнение инференса
        results = model.predict(source=frame, conf=0.3, show=False)

        # Обнуление счетчиков на каждый кадр
        wagon_counts = [{} for _ in range(len(rails_coordinates))]

        # Обработка результатов
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Применяем NMS (Non-Maximum Suppression) для фильтрации дублирующих рамок
                indices = cv2.dnn.NMSBoxes(boxes.xyxy.tolist(), boxes.conf.tolist(), score_threshold=0.3,
                                           nms_threshold=0.3)

                for i in indices.flatten():
                    box = boxes.xyxy[i]
                    x1, y1, x2, y2 = box[:4].numpy()
                    wagon_type = model.names[int(boxes.cls[i])] if int(boxes.cls[i]) < len(
                        model.names) else "Неизвестный тип"

                    # Вычисление центра рамки
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Отображение прямоугольника вокруг вагона
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Проверка пересечения с рельсами
                    intersection_found = False

                    # Перебираем рельсы в прямом и обратном порядке
                    for rail in rails_coordinates[0:2]:
                        j = rails_coordinates.index(rail)
                        intersection_found = False
                        for k in range(len(rail) - 1):
                            rail_start = rail[k]
                            rail_end = rail[k + 1]

                            # Проверяем пересечение
                            if line_intersection((center_x, center_y), (center_x, frame.shape[0]), rail_start,
                                                 rail_end):
                                # Находим точку пересечения
                                intersect_point = intersection_point((center_x, center_y), (center_x, frame.shape[0]),
                                                                     rail_start, rail_end)

                                if intersect_point is not None:
                                    # Если вагон пересекает рельс, увеличиваем счетчик
                                    if wagon_type in wagon_counts[j]:
                                        wagon_counts[j][wagon_type] += 1
                                    else:
                                        wagon_counts[j][wagon_type] = 1
                                    intersection_found = True

                                    # Обрываем перпендикуляр на уровне пересечения
                                    intersect_x, intersect_y = map(int, intersect_point)
                                    cv2.line(frame, (center_x, center_y), (intersect_x, intersect_y), (255, 0, 0),
                                             2)  # Красная линия
                                    break
                        if intersection_found:
                            break

                    for rail in rails_coordinates[::-1][0:5]:
                        j = rails_coordinates.index(rail)
                        intersection_found = False
                        for k in range(len(rail) - 1):
                            rail_start = rail[k]
                            rail_end = rail[k + 1]

                            # Проверяем пересечение
                            if line_intersection((center_x, center_y), (center_x, frame.shape[0]), rail_start,
                                                 rail_end):
                                # Находим точку пересечения
                                intersect_point = intersection_point((center_x, center_y), (center_x, frame.shape[0]),
                                                                     rail_start, rail_end)

                                if intersect_point is not None:
                                    # Если вагон пересекает рельс, увеличиваем счетчик
                                    if wagon_type in wagon_counts[j]:
                                        wagon_counts[j][wagon_type] += 1
                                    else:
                                        wagon_counts[j][wagon_type] = 1
                                    intersection_found = True

                                    # Обрываем перпендикуляр на уровне пересечения
                                    intersect_x, intersect_y = map(int, intersect_point)
                                    cv2.line(frame, (center_x, center_y), (intersect_x, intersect_y), (255, 0, 0),
                                             2)  # Красная линия
                                    break
                        if intersection_found:
                            break
                    # Добавление текста с меткой
                    label = f'{wagon_dict.get(int(wagon_type))} ({int(boxes.conf[i] * 100)}%)'
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                2)

        # Создание полупрозрачного слоя для рельсов
        overlay = frame.copy()
        for rail in rails_coordinates:
            points = np.array(rail, np.int32).reshape((-1, 1, 2))  # Преобразование рельсов в точки
            cv2.polylines(overlay, [points], isClosed=False, color=(255, 0, 0), thickness=2)  # Рисуем рельсы

        # Наложение полупрозрачного слоя на основной кадр
        alpha = 0.2  # Уровень прозрачности
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Отображение счетчиков вагонов на экране с указанием типов (текст красного цвета)
        y_offset = 700
        for i, count_dict in enumerate(wagon_counts):
            text = f"Section {i + 1}: "
            for wagon_type, count in count_dict.items():
                text += f"{wagon_dict.get(int(wagon_type))}: {count}  "  # Сбор информации о количестве вагонов
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                        2)  # Отображение текста (красный цвет)
            y_offset += 30

        # Отображение результата
        cv2.imshow("Tracking", frame)  # Показываем кадр

        # Выход из цикла при нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Освобождение видео
    cv2.destroyAllWindows()  # Закрытие всех окон

# Пример вызова функции
track_and_monitor("video.mp4")
