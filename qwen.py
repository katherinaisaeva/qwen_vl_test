import json
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import re
from datetime import datetime
import gc 
from io import BytesIO
import os

def download_image_from_url(image_url):
    """Загружает изображение по URL"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if max(image.size) > 800:
            image.thumbnail((800, 800))
            
        return image
        
    except Exception as e:
        print(f"Ошибка загрузки изображения по URL {image_url}: {e}")
        return None

def get_image_description_from_url(prompt, image_url):
    """Получает описание изображения по URL"""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        dtype=torch.float16, 
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True 
    )

    image = download_image_from_url(image_url)
    if not image:
        return None
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text], 
        images=[image],
        return_tensors="pt",
        padding=True
    )
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del inputs, generated_ids, image
    gc.collect()

    cleaned_response = re.sub(r'<[^>]*>', '', response).strip()
    
    return cleaned_response

def process_from_txt_urls(input_txt_path, output_json_path):
    """Читает URL-адреса из текстового файла и обрабатывает их"""
    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        results = []
        prompt = "Опиши это изображение подробно на русском языке."
        
        print(f"Найдено {len(urls)} URL-адресов для обработки")
        
        for i, image_url in enumerate(urls, 1):
            print(f"Обработка {i}/{len(urls)}: {image_url}")
            
            description = get_image_description_from_url(prompt, image_url)
            
            if description:
                results.append({
                    "image_url": image_url,
                    "description": description,
                })
                
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"Успешно обработано: {image_url}")
            else:
                print(f"Ошибка при обработке: {image_url}")
        
        return results
            
    except FileNotFoundError:
        print(f"Файл {input_txt_path} не найден")
        return []
    except Exception as e:
        print(f"Ошибка при чтении файла {input_txt_path}: {e}")
        return []

if __name__ == "__main__":
    INPUT_TXT = "/home/eisaeva/projects/qwen_vl/qwen_vl_test/urls.txt"
    OUTPUT_JSON = "image_descriptions.json"
    
    results = process_from_txt_urls(INPUT_TXT, OUTPUT_JSON)
    
    print(f"Готово! Обработано {len(results)} изображений")
    print(f"Результаты сохранены в: {OUTPUT_JSON}")
