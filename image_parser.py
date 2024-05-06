import cv2
import numpy as np
import pytesseract
import json
import os
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def find_regions(image, background_color=(240, 240, 240), top_margin_percent=0, bottom_margin_percent=0, trash_binary_inv=False, canny=False):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисление верхнего и нижнего отступа
    height, _ = gray.shape
    top_margin = int(height * top_margin_percent / 100)
    bottom_margin = int(height * bottom_margin_percent / 100)

    # Применение порогового фильтра для выделения областей с фоном заданного цвета
    _, thresh = cv2.threshold(gray, background_color[0], 255, cv2.THRESH_BINARY_INV if trash_binary_inv else cv2.THRESH_BINARY)

    ## Этот блок задуман для удаления артефактов для более точного вырезания фотографий, еще не сделан
    # Для тестирования смотрим что получилось
    #show_image(trash)

    # Применение алгоритма Кэнни для обнаружения границ
    if canny:
        edges = cv2.Canny(gray, 220, 240, 5)
        # Для тестирования смотрим что получилось
        #show_image(edges)

        # Объединение границ изображения с пороговым изображением
        thresh = cv2.bitwise_or(thresh, edges)
        # Для тестирования смотрим что получилось
        #show_image(tresh)
    ## Конец недоделанного блока

    # Игнорирование верхнего и нижнего отступа
    cropped_thresh = thresh[top_margin:height - bottom_margin, :]

    # Поиск контуров в пороговом изображении
    contours, _ = cv2.findContours(cropped_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Нахождение прямоугольных контуров, которые занимают значительную площадь
    large_rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # Проверка на размер контура и его площадь
        if w > 60 and h > 60 and area > 1000:
            large_rectangles.append((x, y, w, h))

    # Список для хранения координат областей с белым фоном
    regions = []

    # Отображение контуров на исходное изображение
    result = image.copy()

    for rect in large_rectangles:
        x, y, w, h = rect
        # Добавляем смещение для отображения на исходном изображении
        cv2.rectangle(result, (x, y + top_margin), (x + w, y + h + top_margin), (0, 255, 0), 2)
        # Добавляем координаты области в список
        regions.append(((x, y + top_margin), (x + w, y + h + top_margin)))

    return result, regions

def extract_image_region(image, region):
    # Получаем координаты верхнего левого и нижнего правого углов области
    top_left, bottom_right = region
    
    # Вырезаем область из исходного изображения
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    return cropped_image

def is_black_and_white(image, threshold_ratio = 10000):
    x, y, _ = image.shape
    if x * y > threshold_ratio: return False
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Размер изображения
    height, width, channels = blurred.shape
    # Проверяем каждый пиксель изображения
    for y in range(height):
        for x in range(width):
            # Извлекаем цветные каналы пикселя
            b, g, r = blurred[y, x]
            # Проверяем, если цветные каналы не совпадают, то это цветное изображение
            if b != g or b != r or g != r:
                return False
    
    return True

# Функция для изменения размера изображения
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

# Функция для определения наличия текста в области изображения
def has_text(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение порогового фильтра для выделения текста
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Использование Tesseract для распознавания текста
    text = pytesseract.image_to_string(thresh)
    
    # Возвращаем True, если текст обнаружен, иначе False
    return bool(text.strip())

# Функция для сохранения области изображения в файл
def save_image(region, filename):
    cv2.imwrite(filename, region)

# Функция для просмотра вырезанных зон при тестировании
def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Поиск в первый проход - поиск островов
def get_white_regions(image):
    result_image, regions = find_regions(image, background_color=(240, 240, 240), top_margin_percent=6, bottom_margin_percent=6, trash_binary_inv=False)
    # Отображение результата
    # Изменение размера изображения до 50% от исходного
    result_image = resize_image(result_image, 50)
    #show_image(result_image)
    return regions

# Поиск во второй проход - поиск по острову картинок и иконок
def get_large_not_white_regions(image):
    result, regions = find_regions(image, background_color=(240, 240, 240), top_margin_percent=0, bottom_margin_percent=0, trash_binary_inv=True, canny=True)
    # Изменение размера изображения до 50% от исходного
    result = resize_image(result, 50)
    # Для тестирования смотрим что получилось
    #show_image(result)
    return regions

# Сортировка прямоугольников по рядам (от верхних рядов к нижним) слева направо
def sort_rect(array):
    data = array[:]
    result = []
    while len(data) > 1:
        upper_left_rect = get_left_upper_rect(data)
        data.remove(upper_left_rect)
        result.append(upper_left_rect)
    if len(data) > 0:
        result.append(data[0])

    return result

# Поиск самого левого прямоугольника в самом верхнем ряду. Прямоугольник считается в верхним ряду если он своими нижними углами выше верхних углов другого прямоугольника
def get_left_upper_rect(array):
    result = array[:]
    while len(result) > 1:
        sorted_array = list(sorted(result, key=lambda x: x[0][0]))
        left_rect = sorted_array[0]
        lower_rects = list(filter(lambda x: x[1][1] > left_rect[0][1], sorted_array[1:]))
        if len(lower_rects) > 0:
            for el in lower_rects:
                result.remove(el)
        else:
            result.remove(left_rect)
    return result[0]

# Вычисляем площадь прямоугольника
def region_area(region):
    return (region[1][0]-region[0][0])*(region[1][1]-region[0][1])

# Вычисляем соотношение сторон прямоугольника
def aspect_ratio(region):
    width = region[1][0] - region[0][0]
    height = region[1][1] - region[0][1]

    return max(width, height) / min(width, height) if min(width, height) != 0 else float('inf')

# Разделяем прямоугольники на зоны иконок и зоны фотографий, игнорируя обрезки картинок
def split_regions_by_area(regions, threshold_area=10000):
    small_regions = []
    large_regions = []
    
    for region in regions:
        area = region_area(region)        
        if area <= threshold_area:
            small_regions.append(region)
        else:
            large_regions.append(region)
    
    return small_regions, large_regions

# Проверяем вписан ли rect2 в rect1
def is_inside(rect1, rect2):
    x1_rect1, y1_rect1 = rect1[0]
    x2_rect1, y2_rect1 = rect1[1]
    x1_rect2, y1_rect2 = rect2[0]
    x2_rect2, y2_rect2 = rect2[1]
    
    return x1_rect2 >= x1_rect1 and y1_rect2 >= y1_rect1 and x2_rect2 <= x2_rect1 and y2_rect2 <= y2_rect1

# Получаем список прямоугольников в виде пар индексов (индекс внешнего прямоугольника, индекс вписанного прямоугольника)
def find_inside_rectangles(rectangles):
    inside_rectangles = []
    for i, rect1 in enumerate(rectangles):
        for j, rect2 in enumerate(rectangles):
            if i != j and is_inside(rect1, rect2):
                inside_rectangles.append((i, j))
    return inside_rectangles

# Удаляем артефакты картинок (соотношение сторон которых больше threshold_ratio) и решаем какой из пары пересекающихся прямоугольников лишний
def rectangle_fluctuation_filter(rectangles, threshold_ratio = 2.5):    
    first_filtered_rectangles = [rect for rect in rectangles if aspect_ratio(rect) <= threshold_ratio]
    pairs_inscribed_rectangle_indices = find_inside_rectangles(first_filtered_rectangles)
    indices = []
    for el in pairs_inscribed_rectangle_indices:
        if region_area(first_filtered_rectangles[el[0]]) / region_area(first_filtered_rectangles[el[1]]) > 2:
            indices.append(el[1])
        else:
            indices.append(el[0])   

    return [rect for i, rect in enumerate(first_filtered_rectangles) if i not in indices]    

# Обрабатываем картинку
def parse_image(image):
    image_data = []
    # Получаем список регионов изначальной картинки обрамленных белым фоном - т.н. островов
    regions = get_white_regions(image)
    for i, region in enumerate(sort_rect(regions)):
        island_data = []
        # Вырезаем картинку содержащую только остров
        image_island = extract_image_region(image, region)
        # Ищем на острове не белые области - картинки и иконки большого размера (чтобы не захватывать буквы слов)
        not_white_regions = get_large_not_white_regions(image_island)
        # Фильтруем из списка черно белые картинки (иногда в регионах попадается текст, но если увеличить размер регона то раньше чем текст перестанет захватываться исчезают иконки)
        not_white_regions = list(filter(lambda x: not is_black_and_white(extract_image_region(image_island, x)), not_white_regions))
        filtered_not_white_regions = rectangle_fluctuation_filter(not_white_regions, threshold_ratio = 2.5)
        # Разделяем голосовые иконки (маленькие области) и фотографии (большие области)
        small_regions, large_regions = split_regions_by_area(filtered_not_white_regions, threshold_area=10000)
        regions_with_text = []
        # Перебираем регионы со звуковми иконками
        for j, small_region in enumerate(sort_rect(small_regions)):
            # Область распрознавания текста
            text_region = ((None, None), (small_region[0][0], None))
            # Находим ближайший к иконке большой регион снизу
            nearest_down_large_region = list(sorted(large_regions, key=lambda x: (abs(x[1][0]-small_region[1][0]) + abs(x[0][1]-small_region[1][1]))))[0]
            # Отбираем только маленькие регионы слева от иконки
            nearest_left_small_regions = list(filter(lambda x: (x != small_region and (x[1][0] < nearest_down_large_region[0][0])), small_regions))
            # Проверяем есть ли маленькие регионы слева от иконки
            if (len(nearest_left_small_regions) > 0):
                # Если да, то сортируем по близости к текущей иконке
                sorted_nearest_left_small_regions = list(sorted(nearest_left_small_regions, key=lambda x: small_region[0][0]-x[1][1]))
                # Выбираем ближайшую
                point = sorted_nearest_left_small_regions[0]
                # Задаем по x ее правого нижнего угла
                text_region = ((point[1][0], text_region[0][1]), (text_region[1][0], nearest_down_large_region[0][1]))
            else:
                # Если нет, то устанавливаем 0 - край экрана
                text_region = ((0, text_region[0][1]), (text_region[1][0], nearest_down_large_region[0][1]))
            # Находим большие регионы сверху иконки
            up_large_regions = list(filter(lambda x: small_region[0][1] > x[1][1], large_regions))
            if len(up_large_regions) > 0:
                nearest_up_large_region = list(sorted(up_large_regions, key=lambda x: (abs(x[1][0]-small_region[1][0]) + abs(x[0][1]-small_region[1][1]))))[0]
                text_region = ((text_region[0][0], nearest_up_large_region[1][1]), text_region[1])
            else:
                # Значит выше больших областей нет и границей выступит граница экрана т.е. 0
                text_region = ((text_region[0][0], 0), text_region[1])
            # Вырезаем из картинки острова, не белый регион
            image_not_white_region = extract_image_region(image_island, nearest_down_large_region)
            # Вырезаем из картинки область текста
            image_text_region = extract_image_region(image_island, text_region)

            # Для тестирования смотрим что получилось
            #show_image(image_text_region)

            # Распознаем текст
            text = pytesseract.image_to_string(image_text_region)
            text = text.replace('\n', ' ').strip() 
            regions_with_text.append({nearest_down_large_region: {'image': image_not_white_region, 'text': text}})
        
        # Сравниваем регионы из списка с теми что добавлены в словарь
        for j, large_region in enumerate(sort_rect(large_regions)):
            found_dict = None
            for d in regions_with_text:
                if large_region in d:
                    found_dict = d
                    break
            # Формируем окончательный отсортированный список изображений и текстов острова
            if found_dict:
                island_data.append(found_dict[large_region])
            else:
                island_data.append({'image': extract_image_region(image_island, large_region)})
        image_data.append(island_data)
    return image_data

def save_files(data, file_name_prefix):
    json_data = []
    # Путь к папке, в которой вы хотите сохранить изображение
    folder_path = f"{file_name_prefix}" 

    #Проверяем есть ли такая папка и если нет то создаем
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, island in enumerate(data):
        # Комбинация имени обрабатываемого файла и номера острова
        file_prefix = f"image{file_name_prefix}-{i+1:02d}"
        island_data = []
        for j, elem in enumerate(island):
            file_name = f"{file_prefix}-{j+1:02d}.png"
            # Полный путь к файлу
            file_path = folder_path + "\\" + file_name
            print(file_path)
            # Сохранение изображения
            save_image(elem['image'], file_path)
            text = elem.setdefault('text')
            island_data.append({"image": file_name, "text": text})
        json_data.append(island_data)
    
    # Полный путь к JSON файлу
    json_file_path = os.path.join(folder_path, f"{file_name_prefix}.json")
    # Сохранение данных в JSON файл
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def parse_in_folder(folder_path):    

    # Получаем список файлов в папке
    files = os.listdir(folder_path)

    # Фильтруем список, оставляя только файлы (без подпапок)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    for file in files:
        file_path = folder_path + file
        image = cv2.imread(file_path)
        data = parse_image(image)
        file_name = os.path.basename(file)
        save_files(data, file_name.split('.')[0])
        # Сохранение оригинального изображения
        save_image(image, f'{file_name.split('.')[0]}\\{file_name}')
        time.sleep(0.1)

def test():
    file = r'E:\download\samples\002.png'

    image = cv2.imread(file)
    #icon_1_image = cv2.imread(icon_1)
    #icon_2_image = cv2.imread(icon_2)
    #print(icon_1_image.shape, icon_2_image.shape)

    # Изменение размера изображения до 50% от исходного
    #resized_image = resize_image(image, 50)

    data = parse_image(image)
    file_name = os.path.basename(file)
    save_files(data, file_name.split('.')[0])
    # Сохранение оригинального изображения
    save_image(image, f'{file_name.split('.')[0]}\\{file_name}')

folder_path = f'E:\\download\\samples\\'

parse_in_folder(folder_path)
#test()