import requests
import logging
import re
import concurrent.futures
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple
import xml.etree.ElementTree as ET
from collections import Counter
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

@dataclass
class PageInfo:
    url: str
    title: Optional[str] = None
    keywords: Optional[List[str]] = None
    description: Optional[str] = None
    content_type: Optional[str] = None  # 'contact', 'about', 'legal', etc.
    priority: float = 0.0  # Priorité pour l'analyse
    last_modified: Optional[str] = None

class SitemapAnalyzer:
    def __init__(self, max_pages=300, concurrency=5, timeout=15, respect_robots=True):
        self.logger = logging.getLogger(__name__)
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.user_agent = "SitemapAnalyzer/2.0"
        self.headers = {"User-Agent": self.user_agent}
        
        # Mots-clés pour identifier les pages de contact
        self.contact_keywords = [
            # Français
            'contact', 'contacts', 'nous-contacter', 'contactez-nous', 'contactez', 'coordonnées',
            'adresse', 'téléphone', 'telephone', 'email', 'courriel', 'mail', 'formulaire',
            # Anglais
            'get-in-touch', 'reach-us', 'contact-us', 'reach-out', 'help', 'support',
            # À propos
            'propos', 'about', 'about-us', 'a-propos', 'qui-sommes-nous', 'notre-entreprise',
            'our-company', 'notre-equipe', 'our-team', 'our-story', 'history', 'histoire',
            # Légal
            'mentions-legales', 'legal', 'cgu', 'cgv', 'terms', 'conditions', 'privacy'
        ]
        
        # Expressions régulières pour évaluer la pertinence d'une URL
        self.url_patterns = [
            r'(?i)contact|nous[-_]contacter|coordonn[ée]es',
            r'(?i)a[-_]propos|about[-_]us|qui[-_]sommes[-_]nous',
            r'(?i)equipe|team|staff|organization',
            r'(?i)mentions[-_]l[ée]gales|legal|terms|conditions',
        ]
        
    def _get_robots_rules(self, base_url):
        """Récupère les règles du fichier robots.txt"""
        try:
            from urllib.robotparser import RobotFileParser
            robots_url = urljoin(base_url, "/robots.txt")
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.read()
            return parser
        except Exception as e:
            self.logger.warning(f"Erreur lors de la lecture du robots.txt: {e}")
            return None
    
    def _is_allowed(self, url, robots_parser):
        """Vérifie si l'URL est autorisée selon robots.txt"""
        if not self.respect_robots or not robots_parser:
            return True
        return robots_parser.can_fetch(self.user_agent, url)
    
    def _fetch_url(self, url, timeout=None):
        """Récupère le contenu d'une URL avec gestion des erreurs"""
        if timeout is None:
            timeout = self.timeout
            
        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            return response.text, response.headers.get('content-type', '')
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Erreur lors de la récupération de {url}: {e}")
            return None, None
    
    def _is_sitemap_index(self, content):
        """Détermine si le contenu est un index de sitemaps"""
        return '<sitemapindex' in content.lower()
    
    def _parse_sitemap_index(self, content, base_url):
        """Parse un index de sitemaps et retourne les URLs des sitemaps"""
        sitemap_urls = []
        try:
            root = ET.fromstring(content)
            # Tenir compte de l'espace de noms
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            for sitemap in root.findall('.//sm:sitemap', ns) or root.findall('.//sitemap', ns):
                loc_elem = sitemap.find('./sm:loc', ns) or sitemap.find('./loc')
                if loc_elem is not None and loc_elem.text:
                    sitemap_urls.append(loc_elem.text.strip())
            
            if not sitemap_urls:  # Si le parsing avec namespace échoue
                for sitemap in root.findall('.//sitemap'):
                    loc = sitemap.find('./loc')
                    if loc is not None and loc.text:
                        sitemap_urls.append(loc.text.strip())
        except ET.ParseError as e:
            self.logger.error(f"Erreur lors du parsing du sitemap index: {e}")
            
            # Fallback à une approche par regex
            pattern = r'<loc>(.*?)</loc>'
            matches = re.findall(pattern, content)
            for match in matches:
                if '.xml' in match.lower():  # Probablement un sitemap
                    sitemap_urls.append(match)
        
        return sitemap_urls
    
    def _parse_sitemap(self, content, base_url):
        """Parse un sitemap XML et retourne les URLs des pages avec leurs métadonnées"""
        page_infos = []
        try:
            root = ET.fromstring(content)
            # Tenir compte de l'espace de noms
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            for url_elem in root.findall('.//sm:url', ns) or root.findall('.//url', ns):
                loc_elem = url_elem.find('./sm:loc', ns) or url_elem.find('./loc')
                lastmod_elem = url_elem.find('./sm:lastmod', ns) or url_elem.find('./lastmod')
                priority_elem = url_elem.find('./sm:priority', ns) or url_elem.find('./priority')
                
                if loc_elem is not None and loc_elem.text:
                    url = loc_elem.text.strip()
                    lastmod = lastmod_elem.text.strip() if lastmod_elem is not None and lastmod_elem.text else None
                    priority = float(priority_elem.text) if priority_elem is not None and priority_elem.text else 0.5
                    
                    # Déterminer le type de contenu basé sur l'URL
                    content_type = self._infer_content_type_from_url(url)
                    
                    # Ajuster la priorité en fonction du type de contenu
                    if content_type in ['contact', 'about']:
                        priority += 0.3
                    
                    page_infos.append(PageInfo(
                        url=url,
                        last_modified=lastmod,
                        priority=priority,
                        content_type=content_type
                    ))
            
            if not page_infos:  # Si le parsing avec namespace échoue
                # Fallback à une approche par regex
                pattern = r'<url>\s*<loc>(.*?)</loc>(?:.*?<lastmod>(.*?)</lastmod>)?(?:.*?<priority>(.*?)</priority>)?'
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    url = match[0].strip()
                    lastmod = match[1].strip() if match[1] else None
                    priority = float(match[2]) if match[2] else 0.5
                    
                    content_type = self._infer_content_type_from_url(url)
                    if content_type in ['contact', 'about']:
                        priority += 0.3
                    
                    page_infos.append(PageInfo(
                        url=url,
                        last_modified=lastmod,
                        priority=priority,
                        content_type=content_type
                    ))
        except ET.ParseError as e:
            self.logger.error(f"Erreur lors du parsing du sitemap: {e}")
        
        return page_infos
    
    def _infer_content_type_from_url(self, url):
        """Infère le type de contenu à partir de l'URL"""
        url_path = urlparse(url).path.lower()
        url_path = url_path.replace('-', ' ').replace('_', ' ').replace('/', ' ')
        
        # Vérifier les mots-clés dans l'URL
        if any(re.search(rf'\b{re.escape(kw)}\b', url_path) for kw in ['contact', 'nous contacter']):
            return 'contact'
        elif any(re.search(rf'\b{re.escape(kw)}\b', url_path) for kw in ['propos', 'about', 'qui sommes']):
            return 'about'
        elif any(re.search(rf'\b{re.escape(kw)}\b', url_path) for kw in ['legal', 'mentions', 'terms']):
            return 'legal'
        elif any(re.search(rf'\b{re.escape(kw)}\b', url_path) for kw in ['equipe', 'team', 'staff']):
            return 'team'
        
        # Analyse plus poussée avec regex
        for pattern in self.url_patterns:
            if re.search(pattern, url):
                if 'contact' in pattern:
                    return 'contact'
                elif 'propos' in pattern or 'about' in pattern:
                    return 'about'
                elif 'equipe' in pattern or 'team' in pattern:
                    return 'team'
                elif 'legal' in pattern or 'mentions' in pattern:
                    return 'legal'
        
        return 'unknown'
    
    def _discover_sitemaps(self, base_url):
        """Découvre tous les sitemaps d'un site"""
        all_sitemaps = set()
        potential_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemaps.xml',
            '/sitemap/sitemap.xml',
            '/sitemap/index.xml',
            '/sitemap1.xml',
            '/wp-sitemap.xml',  # WordPress
            '/sitemap/sitemap-index.xml',
        ]
        
        # Essayer d'abord robots.txt
        robots_content, _ = self._fetch_url(urljoin(base_url, '/robots.txt'))
        if robots_content:
            sitemap_matches = re.findall(r'(?i)Sitemap:\s*(https?://[^\s]+)', robots_content)
            for match in sitemap_matches:
                all_sitemaps.add(match)
        
        # Essayer les chemins communs
        for path in potential_paths:
            sitemap_url = urljoin(base_url, path)
            content, _ = self._fetch_url(sitemap_url)
            if content and ('<urlset' in content.lower() or '<sitemapindex' in content.lower()):
                all_sitemaps.add(sitemap_url)
        
        return list(all_sitemaps)
    
    def _get_all_sitemap_urls(self, base_url):
        """Récupère toutes les URLs des pages à partir de tous les sitemaps"""
        sitemap_urls = self._discover_sitemaps(base_url)
        
        if not sitemap_urls:
            self.logger.warning(f"Aucun sitemap trouvé pour {base_url}")
            return []
        
        all_page_infos = []
        processed_sitemaps = set()
        sitemaps_to_process = list(sitemap_urls)
        
        while sitemaps_to_process:
            sitemap_url = sitemaps_to_process.pop(0)
            if sitemap_url in processed_sitemaps:
                continue
                
            processed_sitemaps.add(sitemap_url)
            content, _ = self._fetch_url(sitemap_url)
            
            if not content:
                continue
                
            if self._is_sitemap_index(content):
                child_sitemaps = self._parse_sitemap_index(content, base_url)
                sitemaps_to_process.extend(child_sitemaps)
            else:
                page_infos = self._parse_sitemap(content, base_url)
                all_page_infos.extend(page_infos)
        
        return all_page_infos
    
    def _analyze_page_for_contacts(self, page_info):
        """Analyse une page pour en extraire des informations de contact"""
        content, content_type = self._fetch_url(page_info.url)
        if not content:
            return None
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Récupérer le titre
            title_tag = soup.find('title')
            page_info.title = title_tag.text.strip() if title_tag else None
            
            # Récupérer les méta-keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords and meta_keywords.get('content'):
                page_info.keywords = [k.strip() for k in meta_keywords['content'].split(',')]
            
            # Récupérer la méta-description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            page_info.description = meta_desc['content'] if meta_desc and meta_desc.get('content') else None
            
            # Affiner le type de contenu
            if page_info.content_type == 'unknown':
                page_info.content_type = self._infer_content_type_from_content(soup, page_info.url)
            
            return page_info
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de {page_info.url}: {e}")
            return None
    
    def _infer_content_type_from_content(self, soup, url):
        """Infère le type de contenu à partir du contenu de la page"""
        # Vérifier le titre
        title = soup.find('title')
        if title:
            title_text = title.text.lower()
            if any(kw in title_text for kw in ['contact', 'nous contacter']):
                return 'contact'
            elif any(kw in title_text for kw in ['propos', 'about us', 'qui sommes']):
                return 'about'
        
        # Vérifier les h1, h2
        headers = soup.find_all(['h1', 'h2'])
        for header in headers:
            header_text = header.text.lower()
            if any(kw in header_text for kw in ['contact', 'nous contacter']):
                return 'contact'
            elif any(kw in header_text for kw in ['propos', 'about us', 'qui sommes']):
                return 'about'
        
        # Vérifier le contenu général
        body_text = soup.get_text().lower()
        contact_score = sum(body_text.count(kw) for kw in ['contact', 'email', 'téléphone', 'adresse'])
        about_score = sum(body_text.count(kw) for kw in ['propos', 'about', 'histoire', 'mission'])
        
        if contact_score > about_score and contact_score > 3:
            return 'contact'
        elif about_score > contact_score and about_score > 3:
            return 'about'
        
        return 'unknown'
    
    def _cluster_pages(self, page_infos, n_clusters=5):
        """Cluster les pages par similarité pour identifier des groupes thématiques"""
        # Extraire les URL paths pour clustering
        urls = [urlparse(p.url).path for p in page_infos]
        
        # Vectorisation TF-IDF des URL paths
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 6))
        X = vectorizer.fit_transform(urls)
        
        # Clustering K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Associer chaque page à son cluster
        for i, page_info in enumerate(page_infos):
            page_info.cluster = int(clusters[i])
        
        # Analyser chaque cluster pour déterminer son thème dominant
        cluster_themes = {}
        for cluster_id in range(n_clusters):
            cluster_pages = [p for i, p in enumerate(page_infos) if clusters[i] == cluster_id]
            cluster_paths = [urlparse(p.url).path for p in cluster_pages]
            
            # Analyser les mots communs
            all_words = []
            for path in cluster_paths:
                words = re.findall(r'[a-zA-Z]+', path.lower())
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(5)
            cluster_themes[cluster_id] = [word for word, _ in top_words]
        
        return page_infos, cluster_themes
    
    def find_important_pages(self, base_url, max_analysis=50):
        """Trouve les pages importantes à partir du sitemap avec une analyse avancée"""
        # Initialiser le parser de robots.txt
        robots_parser = self._get_robots_rules(base_url) if self.respect_robots else None
        
        # Récupérer toutes les pages du sitemap
        all_pages = self._get_all_sitemap_urls(base_url)
        
        if not all_pages:
            self.logger.warning(f"Aucune page trouvée dans les sitemaps pour {base_url}")
            # Tentative de découverte par crawl de la page d'accueil
            return self._discover_pages_by_crawling(base_url, robots_parser)
        
        # Filtrer les pages autorisées par robots.txt
        if self.respect_robots:
            all_pages = [p for p in all_pages if self._is_allowed(p.url, robots_parser)]
        
        # Trier par priorité (type de contenu + priorité de sitemap)
        all_pages.sort(key=lambda p: p.priority, reverse=True)
        
        # Limiter le nombre de pages à analyser
        pages_to_analyze = all_pages[:min(max_analysis, len(all_pages))]
        
        # Analyser les pages en parallèle
        analyzed_pages = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_page = {executor.submit(self._analyze_page_for_contacts, page): page 
                             for page in pages_to_analyze}
            
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    result = future.result()
                    if result:
                        analyzed_pages.append(result)
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'analyse de {page.url}: {e}")
        
        # Appliquer le clustering pour grouper les pages similaires
        analyzed_pages, cluster_themes = self._cluster_pages(analyzed_pages)
        
        # Extraire les pages importantes par type
        contact_pages = [p for p in analyzed_pages if p.content_type == 'contact']
        about_pages = [p for p in analyzed_pages if p.content_type == 'about']
        
        # Si aucune page de contact trouvée, essayer de trouver dans d'autres clusters
        if not contact_pages:
            # Chercher des clusters qui peuvent contenir des pages de contact
            potential_contact_clusters = []
            for cluster_id, themes in cluster_themes.items():
                if any(theme in self.contact_keywords for theme in themes):
                    potential_contact_clusters.append(cluster_id)
            
            # Ajouter les pages de ces clusters
            for p in analyzed_pages:
                if hasattr(p, 'cluster') and p.cluster in potential_contact_clusters:
                    contact_pages.append(p)
        
        return {
            'contact_pages': contact_pages,
            'about_pages': about_pages,
            'all_analyzed': analyzed_pages,
            'cluster_themes': cluster_themes
        }
    
    def _discover_pages_by_crawling(self, base_url, robots_parser=None, max_depth=2):
        """Découvre les pages importantes par crawl si aucun sitemap n'est disponible"""
        self.logger.info(f"Découverte de pages par crawl pour {base_url}")
        
        discovered_pages = []
        visited = set()
        to_visit = [(base_url, 0)]  # (url, depth)
        
        while to_visit and len(discovered_pages) < self.max_pages:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
                
            visited.add(url)
            
            # Vérifier robots.txt
            if robots_parser and not self._is_allowed(url, robots_parser):
                continue
                
            content, _ = self._fetch_url(url)
            if not content:
                continue
                
            # Analyser la page
            content_type = self._infer_content_type_from_url(url)
            page_info = PageInfo(url=url, content_type=content_type)
            
            if content_type in ['contact', 'about']:
                page_info.priority = 0.9
            elif depth == 0:  # Page d'accueil
                page_info.priority = 0.8
            else:
                page_info.priority = 0.5 - (depth * 0.1)
                
            discovered_pages.append(page_info)
            
            # Ne pas continuer à crawler si profondeur max atteinte
            if depth >= max_depth:
                continue
                
            # Extraire les liens pour le crawling
            try:
                soup = BeautifulSoup(content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link.get('href', '').strip()
                    if not href or href.startswith('#') or href.startswith('javascript:'):
                        continue
                        
                    # Normaliser l'URL
                    next_url = urljoin(url, href)
                    
                    # S'assurer que l'URL est sur le même domaine
                    if urlparse(next_url).netloc != urlparse(base_url).netloc:
                        continue
                        
                    # Prioriser les liens pertinents
                    if any(kw in href.lower() for kw in self.contact_keywords):
                        to_visit.insert(0, (next_url, depth + 1))  # Priorité haute
                    else:
                        to_visit.append((next_url, depth + 1))
            except Exception as e:
                self.logger.error(f"Erreur lors de l'extraction des liens de {url}: {e}")
        
        # Analyser les pages découvertes
        analyzed_pages = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_page = {executor.submit(self._analyze_page_for_contacts, page): page 
                             for page in discovered_pages}
            
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    result = future.result()
                    if result:
                        analyzed_pages.append(result)
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'analyse de {page.url}: {e}")
        
        # Extraire les pages importantes par type
        contact_pages = [p for p in analyzed_pages if p.content_type == 'contact']
        about_pages = [p for p in analyzed_pages if p.content_type == 'about']
        
        return {
            'contact_pages': contact_pages,
            'about_pages': about_pages,
            'all_analyzed': analyzed_pages,
        }

# Exemple d'utilisation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = SitemapAnalyzer(max_pages=100, concurrency=10)
    # results = analyzer.find_important_pages("https://www.groupe-pomona.fr/")
    results = analyzer.find_important_pages("https://www.optimail-solutions.com/")
    
    print("\n=== PAGES DE CONTACT ===")
    for page in results['contact_pages']:
        print(f"URL: {page.url}")
        print(f"Titre: {page.title}")
        print(f"Type: {page.content_type}")
        print(f"Priorité: {page.priority}")
        print("-" * 50)
    
    print("\n=== PAGES À PROPOS ===")
    for page in results['about_pages']:
        print(f"URL: {page.url}")
        print(f"Titre: {page.title}")
        print(f"Type: {page.content_type}")
        print(f"Priorité: {page.priority}")
        print("-" * 50)
    
    if 'cluster_themes' in results:
        print("\n=== THÈMES DE CLUSTERS ===")
        for cluster_id, themes in results['cluster_themes'].items():
            print(f"Cluster {cluster_id}: {', '.join(themes)}")