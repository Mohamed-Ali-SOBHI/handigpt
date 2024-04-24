import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder


def clean_text(text):
    """Clean text by removing extra spaces and HTML tags."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def read_and_segment_file(file_path):
    """Read a text file and extract non-empty text blocks separated by blank lines, cleaning them."""
    with open(file_path, 'r', encoding='utf-8') as file:
        segments = [clean_text(segment) for segment in file.read().split('\n\n') if segment.strip()]
    return segments

def clean_and_filter_segments(segments, min_length=20):
    """Clean and filter segments to exclude non-informative ones."""
    exclude_phrases = ["Temps de lecture :", "Mis à jour le"]
    return [segment for segment in segments if len(segment.split()) >= min_length and not any(phrase in segment for phrase in exclude_phrases)]

def group_short_segments(segments, min_length=20):
    """Group short segments into longer ones to provide more context."""
    grouped_segments, current_segment = [], ""
    for segment in segments:
        if len(current_segment.split()) + len(segment.split()) < min_length:
            current_segment += " " + segment
        else:
            if current_segment: grouped_segments.append(current_segment.strip())
            current_segment = segment
    if current_segment: grouped_segments.append(current_segment.strip())
    return grouped_segments

def generate_questions_and_save_json(segments, output_file, num_questions=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        HfFolder.save_token('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')
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        #tokenizer.pad_token = tokenizer.eos_token
        qa_pairs = []

        for segment in tqdm(segments[:10]):
            prompt = f"""Basé sur le texte suivant, générer des questions informatives, engageantes et bien formulées en français qui pourraient être posées dans un contexte 
                        éducatif ou journalistique. Les questions doivent inciter à la réflexion et à la discussion, en explorant des aspects clés et des détails intéressants 
                        du texte. Texte : "{segment}" """
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(**inputs, max_length=512, num_return_sequences=num_questions, num_beams=5, temperature=0.9)
            
            questions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            qa_pairs.append([{"instruction": question, "output": segment} for question in questions])
        
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(qa_pairs, file, ensure_ascii=False, indent=4)

    except RuntimeError as e:
        print(f"Runtime error during model operation: {e}")

segments = read_and_segment_file('collected_text_data_multi_threaded_1.txt')
cleaned_and_filtered_segments = clean_and_filter_segments(segments)
grouped_segments = group_short_segments(cleaned_and_filtered_segments)
print(f"Number of segments before grouping: {len(cleaned_and_filtered_segments)}")
print(f"Number of segments after grouping: {len(grouped_segments)}")
print(grouped_segments[-1])
generate_questions_and_save_json(grouped_segments, 'generated_questions.json')
