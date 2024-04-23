import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
import threading
import urllib.robotparser

lock = threading.Lock()

def is_allowed(url: str, user_agent='*') -> bool:
    """
    Vérifie si l'accès à une URL est autorisé par robots.txt.
    """
    parser = urllib.robotparser.RobotFileParser()
    robot_txt_url = urlparse(url)._replace(path='/robots.txt', query='').geturl() # Construit l'URL de robots.txt pour le domaine donné
    parser.set_url(robot_txt_url)
    parser.read()
    return parser.can_fetch(user_agent, url)

def fetch_links_and_data_from_web_page(url: str):
    """
    Extrait les liens et le texte d'une page web, en respectant robots.txt.
    """
    if not is_allowed(url):
        print(f"L'accès à {url} est interdit par robots.txt.")
        return [], []

    try:
        response = requests.get(url)
        page_content = response.content
        soup = BeautifulSoup(page_content, 'html.parser')

        paragraphs = soup.find_all('p')
        text_data = [para.get_text() for para in paragraphs]

        links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True) if a.get('href').startswith(('http', 'https'))]

        return links, text_data
    except Exception as e:
        print(f"Erreur lors de la récupération de {url}: {e}")
        return [], []

def save_text_data(text_data):
    """
    Enregistre les données textuelles de manière thread-safe.
    """
    with lock:
        with open("collected_text_data_multi_threaded.txt", "a", encoding='utf-8') as file:
            for line in text_data:
                file.write(f"{line}\n")

def recursive_fetch_content(url: str, visited_urls: set, depth: int = 1):
    """
    Récupère le contenu textuel de manière récursive avec multi-threading, en respectant robots.txt. 
    """
    if depth == 0 or url in visited_urls:
        return
    visited_urls.add(url)

    links, text_data = fetch_links_and_data_from_web_page(url)
    save_text_data(text_data)

    if depth > 1:
        with ThreadPoolExecutor(max_workers=5) as executor:
            for link in links:
                if link not in visited_urls and is_allowed(link):
                    executor.submit(recursive_fetch_content, link, visited_urls, depth-1) # Appel récursif avec multi-threading pour chaque lien trouvé sur la page
                    print(link)
                    
if __name__ == "__main__":
    visited_urls_set = set()
    starting_url = "https://www.monparcourshandicap.gouv.fr/aides"  
    recursive_fetch_content(starting_url, visited_urls_set, depth=4) # Profondeur de 2 pour l'exemple
    print(visited_urls_set)
    print(len(visited_urls_set))
