from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
import os

def add_glossy_highlight(img, intensity=0.25):
    gloss = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(gloss)
    w, h = img.size
    for i in range(int(h*0.5)):
        alpha = int(255 * (intensity * (1 - i/(h*0.5))))
        draw.line([(0,i),(w,i)], fill=(255,255,255,alpha))
    return Image.alpha_composite(img, gloss)

def process_image(input_path, output_path, output_size=(1200,1200), alpha_threshold=80, padding_ratio=0.05):
    # Обрезка
    img = Image.open(input_path).convert("RGBA")
    img = img.rotate(-90, expand=True) # Поворачиваем фото
    session = new_session("u2net")
    obj = remove(img, session=session)

    arr = np.array(obj)
    alpha = arr[:,:,3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        print(f"Объект не найден: {input_path}")
        return
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    obj_crop = obj.crop((min_x, min_y, max_x+1, max_y+1))

    # Масштабирование с отступами
    obj_w, obj_h = obj_crop.size
    max_w = output_size[0] * (1 - 2 * padding_ratio)
    max_h = output_size[1] * (1 - 2 * padding_ratio)
    scale = min(max_w / obj_w, max_h / obj_h)
    new_w, new_h = int(obj_w * scale), int(obj_h * scale)
    obj_resized = obj_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Удаление окантовки
    arr = np.array(obj_resized).astype(np.float32)
    alpha_channel = arr[:,:,3:4] / 255.0
    arr[:,:,:3] = arr[:,:,:3] * alpha_channel + 255 * (1 - alpha_channel)
    obj_resized = Image.fromarray(arr.astype(np.uint8), mode=None)

    # Цветокоррекция
    obj_resized = ImageEnhance.Brightness(obj_resized).enhance(1.4)
    obj_resized = ImageEnhance.Contrast(obj_resized).enhance(1.4)
    obj_resized = ImageEnhance.Color(obj_resized).enhance(1.5)
    obj_resized = ImageEnhance.Sharpness(obj_resized).enhance(1.5)

    # Эффект глянца
    obj_resized = add_glossy_highlight(obj_resized, intensity=0.3)

    # Центрирование с отступами
    bg = Image.new("RGBA", output_size, (255,255,255,255))
    start_x = (output_size[0] - obj_resized.size[0]) // 2
    start_y = (output_size[1] - obj_resized.size[1]) // 2
    bg.paste(obj_resized, (start_x, start_y), mask=obj_resized.split()[3])

    # Сохранение
    bg.convert("RGB").save(output_path, "JPEG", quality=95)
    print(f"Сохранено: {output_path}")

# Пакетная обработка
input_folder = "input_images"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg",".jpeg",".png")):
        process_image(os.path.join(input_folder, file),
                      os.path.join(output_folder, file))
