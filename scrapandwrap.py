import logging
import re
import time
import random
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
import concurrent.futures

@dataclass
class Contact:
    siret: Optional[str] = None
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None
    social_media: Dict[str, str] = field(default_factory=dict)  # Plateforme -> URL
    address: Optional[str] = None
    website: Optional[str] = None
    form_url: Optional[str] = None  # URL du formulaire de contact
    source: str = "website"  # website, gmaps, linkedin, etc.
    confidence: float = 0.0

class ContactFinderPro:
    def __init__(self, timeout=15, max_threads=5, use_proxies=False):
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.max_threads = max_threads
        self.use_proxies = use_proxies
        self.user_agent = "ContactFinder/3.0"
        self.headers = {"User-Agent": self.user_agent}
        
        # Expression régulières pour différents formats d'email
        self.email_patterns = [
            r'[\w.+-]+@[\w-]+\.[\w.-]+',  # Standard: exemple@domaine.com
            r'[\w.+-]+\s*[\(\[]at[\)\]]\s*[\w-]+\.[\w.-]+',  # Format "at"
            r'[\w.+-]+\s*\[\s*@\s*\]\s*[\w-]+\.[\w.-]+',  # Format spécial
            r'[\w.+-]+\s*\(arobase\)\s*[\w-]+\.[\w.-]+',  # Format français
        ]
        
        # Expression régulières pour téléphones
        self.phone_patterns = [
            # Format français
            r'(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}',
            # Format international général
            r'(?:\+\d{1,3}|\(\d{1,3}\))\s*[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        ]
        
        # URLs des réseaux sociaux
        self.social_media_patterns = {
            'linkedin': r'linkedin\.com/(?:company|in)/[^/\s"\']+',
            'facebook': r'facebook\.com/[^/\s"\'?]+',
            'twitter': r'twitter\.com/[^/\s"\'?]+',
            'instagram': r'instagram\.com/[^/\s"\'?]+',
        }
        
        # API keys (à configurer)
        self.api_keys = {
            'google_maps': 'YOUR_GOOGLE_MAPS_API_KEY',
            'societe_info': 'YOUR_SOCIETE_INFO_API_KEY',
            'pappers': 'YOUR_PAPPERS_API_KEY',
        }
        
        # Configuration des proxies
        self.proxies = []
        if self.use_proxies:
            self._load_proxies()
    
    def _load_proxies(self):
        """Charge une liste de proxies (à implémenter selon votre source)"""
        # Exemple avec une liste statique
        self.proxies = [
            "http://proxy1:8080",
            "http://proxy2:8080",
            "http://proxy3:8080",
        ]
    
    def _get_random_proxy(self):
        """Retourne un proxy aléatoire de la liste"""
        if not self.proxies:
            return None
        return random.choice(self.proxies)
    
    def _get_request_params(self):
        """Prépare les paramètres pour une requête HTTP"""
        params = {
            'headers': self.headers,
            'timeout': self.timeout
        }
        
        if self.use_proxies and self.proxies:
            proxy = self._get_random_proxy()
            if proxy:
                params['proxies'] = {'http': proxy, 'https': proxy}
        
        return params
    
    def _fetch_url(self, url):
        """Récupère le contenu d'une URL avec gestion des erreurs et rotation de proxies"""
        try:
            # Délai aléatoire pour éviter d'être bloqué
            time.sleep(random.uniform(0.5, 2.0))
            
            params = self._get_request_params()
            response = requests.get(url, **params)
            response.raise_for_status()
            
            # Détecter l'encodage
            if 'charset' in response.headers.get('content-type', '').lower():
                response.encoding = response.apparent_encoding
                
            return response.text, response.headers.get('content-type', '')
        except Exception as e:
            self.logger.warning(f"Erreur lors de la récupération de {url}: {e}")
            
            # Si on utilise des proxies et qu'il y a eu une erreur, essayer avec un autre proxy
            if self.use_proxies and self.proxies and "proxy" in str(e).lower():
                try:
                    params = self._get_request_params()  # Nouveau proxy
                    response = requests.get(url, **params)
                    response.raise_for_status()
                    return response.text, response.headers.get('content-type', '')
                except Exception as e2:
                    self.logger.error(f"Échec avec proxy alternatif pour {url}: {e2}")
            
            return None, None
    
    def extract_contacts_from_page(self, url, priority=0.0):
        """Extrait les contacts d'une page web"""
        self.logger.info(f"Extraction des contacts de {url} (priorité: {priority})")
        
        content, content_type = self._fetch_url(url)
        if not content:
            return None
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Initialiser l'objet Contact
            contact = Contact(
                website=url,
                source="website",
                confidence=max(0.3, min(0.9, priority))  # Confiance de base liée à la priorité
            )
            
            # 1. Extraire le nom de l'entreprise
            company_name = self._extract_company_name(soup, url)
            if company_name:
                contact.company_name = company_name
            
            # 2. Extraire les emails
            emails = self._extract_emails(soup)
            if emails:
                # Prendre l'email le plus pertinent
                contact.email = self._select_best_email(emails)
                contact.confidence += 0.2
            
            # 3. Extraire les téléphones
            phones = self._extract_phones(soup)
            if phones:
                contact.phone = phones[0]  # Prendre le premier téléphone
                contact.confidence += 0.1
            
            # 4. Extraire les réseaux sociaux
            social_media = self._extract_social_media(soup)
            if social_media:
                contact.social_media = social_media
                contact.confidence += 0.1
            
            # 5. Vérifier s'il y a un formulaire de contact
            contact_form = self._find_contact_form(soup)
            if contact_form:
                contact.form_url = url
                contact.confidence += 0.05
            
            # 6. Extraire l'adresse
            address = self._extract_address(soup)
            if address:
                contact.address = address
                contact.confidence += 0.1
            
            return contact
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de {url}: {e}")
            return None
    
    def _extract_company_name(self, soup, url):
        """Extrait le nom de l'entreprise"""
        # Depuis les méta-tags
        og_site_name = soup.find('meta', property='og:site_name')
        if og_site_name and og_site_name.get('content'):
            return og_site_name['content']
        
        # Depuis le titre
        title = soup.find('title')
        if title:
            title_text = title.get_text().strip()
            
            # Nettoyer le titre
            common_suffixes = [
                '- Contact', '| Contact', '- Nous contacter', '| Nous contacter',
                '- Accueil', '| Accueil', '- Home', '| Home'
            ]
            
            for suffix in common_suffixes:
                if suffix in title_text:
                    parts = title_text.split(suffix)
                    return parts[0].strip()
            
            # Si le titre est court, c'est probablement le nom de l'entreprise
            if len(title_text.split()) <= 5:
                return title_text
        
        # Depuis l'URL
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Convertir domain.com en "Domain"
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            return domain_parts[0].capitalize()
        
        return None
    
    def _extract_emails(self, soup):
        """Extrait tous les emails d'une page"""
        emails = set()
        
        # 1. Recherche dans les balises mailto:
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if href.startswith('mailto:'):
                email = href[7:].strip()  # Enlever 'mailto:'
                
                # Nettoyer l'email (enlever les paramètres)
                if '?' in email:
                    email = email.split('?')[0]
                
                if self._is_valid_email(email):
                    emails.add(email)
        
        # 2. Recherche dans le texte
        page_text = soup.get_text()
        for pattern in self.email_patterns:
            found = re.findall(pattern, page_text, re.IGNORECASE)
            for email in found:
                # Nettoyage des formats spéciaux
                clean_email = email.replace(' ', '').replace('[at]', '@')
                clean_email = re.sub(r'\[\s*@\s*\]', '@', clean_email)
                clean_email = re.sub(r'[\(\[]at[\)\]]', '@', clean_email)
                clean_email = re.sub(r'\(arobase\)', '@', clean_email)
                
                if self._is_valid_email(clean_email):
                    emails.add(clean_email)
        
        # 3. Recherche dans des attributs data-*
        for tag in soup.find_all(attrs={"data-email": True}):
            email = tag.get("data-email")
            if self._is_valid_email(email):
                emails.add(email)
        
        # 4. Recherche dans le code source pour les emails obfusqués
        html_text = str(soup)
        
        # Rechercher des modèles comme "user" + "@" + "domain.com"
        js_email_pattern = r'["\']([\w.+-]+)["\'][\s+]*\+[\s+]*["\']@["\']\s*\+\s*["\']([\w.-]+)["\']'
        js_matches = re.findall(js_email_pattern, html_text)
        
        for match in js_matches:
            if len(match) == 2:
                email = f"{match[0]}@{match[1]}"
                if self._is_valid_email(email):
                    emails.add(email)
        
        return list(emails)
    
    def _is_valid_email(self, email):
        """Vérifie si un email est valide"""
        if not email or '@' not in email:
            return False
            
        # Vérification de base avec regex
        if not re.match(r'^[\w.+-]+@[\w-]+\.[\w.-]+$', email):
            return False
            
        # Éviter les emails trop longs
        if len(email) > 50:
            return False
            
        # Éviter les emails génériques
        generic_emails = ['example', 'test', 'user', 'info@example', 'webmaster@example']
        if any(generic in email.lower() for generic in generic_emails):
            return False
            
        return True
    
    def _select_best_email(self, emails):
        """Sélectionne le meilleur email parmi plusieurs"""
        if not emails:
            return None
            
        if len(emails) == 1:
            return emails[0]
            
        # Priorité selon le type d'email
        email_priority = {
            'contact': 10,
            'info': 9,
            'hello': 8,
            'bonjour': 8,
            'commercial': 7,
            'sales': 7,
            'support': 6,
            'service': 6,
            'direction': 5,
            'admin': 4,
            'webmaster': 3,
            'noreply': 1,
            'no-reply': 1
        }
        
        # Scorer chaque email
        scored_emails = []
        for email in emails:
            score = 5  # Score par défaut
            
            # Vérifier les préfixes connus
            prefix = email.split('@')[0].lower()
            for key, value in email_priority.items():
                if key in prefix:
                    score = value
                    break
            
            scored_emails.append((email, score))
        
        # Retourner l'email avec le score le plus élevé
        scored_emails.sort(key=lambda x: x[1], reverse=True)
        return scored_emails[0][0]
    
    def _extract_phones(self, soup):
        """Extrait tous les téléphones d'une page"""
        phones = set()
        
        # 1. Recherche dans les balises tel:
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if href.startswith('tel:'):
                phone = href[4:].strip()  # Enlever 'tel:'
                phones.add(self._clean_phone(phone))
        
        # 2. Recherche dans le texte
        page_text = soup.get_text()
        for pattern in self.phone_patterns:
            found = re.findall(pattern, page_text)
            for phone in found:
                phones.add(self._clean_phone(phone))
        
        # 3. Recherche dans les attributs spécifiques
        for tag in soup.find_all(attrs={"data-phone": True}):
            phone = tag.get("data-phone")
            phones.add(self._clean_phone(phone))
        
        # Filtrer les téléphones valides
        valid_phones = [p for p in phones if self._is_valid_phone(p)]
        return valid_phones
    
    def _clean_phone(self, phone):
        """Nettoie un numéro de téléphone"""
        # Supprimer tous les caractères non numériques sauf + et ()
        cleaned = re.sub(r'[^\d+()]', '', phone)
        
        # Standardisation des formats français
        if cleaned.startswith('0033'):
            cleaned = '+33' + cleaned[4:]
        elif cleaned.startswith('33') and not cleaned.startswith('+33'):
            cleaned = '+33' + cleaned[2:]
        
        return cleaned
    
    def _is_valid_phone(self, phone):
        """Vérifie si un numéro de téléphone est valide"""
        if not phone:
            return False
            
        # Doit contenir au moins 8 chiffres
        digits = re.sub(r'\D', '', phone)
        if len(digits) < 8:
            return False
            
        return True
    
    def _extract_social_media(self, soup):
        """Extrait les liens vers les réseaux sociaux"""
        social_links = {}
        
        # Rechercher dans tous les liens
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').strip()
            
            # Ignorer les liens vides ou JavaScript
            if not href or href.startswith('javascript:') or href == '#':
                continue
                
            # Vérifier chaque plateforme
            for platform, pattern in self.social_media_patterns.items():
                if re.search(pattern, href, re.IGNORECASE):
                    social_links[platform] = href
                    break
        
        return social_links
    
    def _find_contact_form(self, soup):
        """Détecte la présence d'un formulaire de contact"""
        # 1. Chercher les formulaires explicites
        for form in soup.find_all('form'):
            # Vérifier les attributs du formulaire
            form_id = form.get('id', '').lower()
            form_class = ' '.join(form.get('class', [])).lower()
            form_action = form.get('action', '').lower()
            
            # Vérifier si c'est un formulaire de contact
            contact_indicators = ['contact', 'message', 'feedback', 'enquiry', 'enquête']
            
            if any(ind in form_id for ind in contact_indicators) or \
               any(ind in form_class for ind in contact_indicators) or \
               any(ind in form_action for ind in contact_indicators):
                return True
            
            # Vérifier les champs du formulaire
            inputs = form.find_all(['input', 'textarea'])
            input_names = [inp.get('name', '').lower() for inp in inputs if inp.get('name')]
            input_ids = [inp.get('id', '').lower() for inp in inputs if inp.get('id')]
            input_placeholders = [inp.get('placeholder', '').lower() for inp in inputs if inp.get('placeholder')]
            
            contact_field_indicators = ['email', 'mail', 'name', 'nom', 'message', 'subject', 'sujet']
            
            if any(ind in ''.join(input_names) for ind in contact_field_indicators) or \
               any(ind in ''.join(input_ids) for ind in contact_field_indicators) or \
               any(ind in ''.join(input_placeholders) for ind in contact_field_indicators):
                return True
        
        # 2. Chercher des div qui pourraient contenir des formulaires
        contact_divs = soup.find_all(['div', 'section'], id=lambda x: x and 'contact' in x.lower())
        contact_divs.extend(soup.find_all(['div', 'section'], class_=lambda x: x and 'contact' in x.lower()))
        
        for div in contact_divs:
            if div.find(['input', 'textarea', 'button', 'form']):
                return True
        
        return False
    
    def _extract_address(self, soup):
        """Extrait l'adresse de l'entreprise"""
        # 1. Chercher dans les données structurées
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    for item in data:
                        address = self._extract_address_from_jsonld(item)
                        if address:
                            return address
                else:
                    address = self._extract_address_from_jsonld(data)
                    if address:
                        return address
            except Exception as e:
                self.logger.debug(f"Erreur lors du parsing JSON-LD: {e}")
        
        # 2. Chercher dans des balises dédiées
        address_elements = soup.find_all(['div', 'p', 'span'], 
                                       class_=lambda x: x and any(word in x.lower() for word in ['address', 'adresse', 'coordonnées']))
        address_elements.extend(soup.find_all(['div', 'p', 'span'], 
                                           id=lambda x: x and any(word in x.lower() for word in ['address', 'adresse', 'coordonnées'])))
        
        for element in address_elements:
            text = element.get_text().strip()
            if len(text) > 10 and re.search(r'\d+\s+\w+', text):  # Recherche de format d'adresse basique
                return text
        
        return None
    
    def _extract_address_from_jsonld(self, data):
        """Extrait l'adresse à partir de données JSON-LD"""
        if not isinstance(data, dict):
            return None
            
        # Schema.org Organization ou LocalBusiness
        if '@type' in data and data['@type'] in ['Organization', 'LocalBusiness', 'Store', 'Restaurant']:
            if 'address' in data and isinstance(data['address'], dict):
                addr = data['address']
                address_parts = []
                
                if 'streetAddress' in addr:
                    address_parts.append(addr['streetAddress'])
                    
                if 'postalCode' in addr:
                    address_parts.append(addr['postalCode'])
                    
                if 'addressLocality' in addr:
                    address_parts.append(addr['addressLocality'])
                    
                if 'addressCountry' in addr:
                    country = addr['addressCountry']
                    if isinstance(country, dict) and 'name' in country:
                        address_parts.append(country['name'])
                    else:
                        address_parts.append(country)
                    
                if address_parts:
                    return ', '.join(address_parts)
        
        return None
    
    def _find_contact_info_from_gmaps(self, company_name, address=None):
        """Recherche les informations de contact via Google Maps API"""
        if not self.api_keys.get('google_maps') or self.api_keys.get('google_maps') == 'YOUR_GOOGLE_MAPS_API_KEY':
            self.logger.warning("Clé API Google Maps non configurée")
            return None
            
        try:
            # Construire la requête
            search_query = company_name
            if address:
                search_query += f" {address}"
                
            params = {
                'key': self.api_keys['google_maps'],
                'input': search_query,
                'inputtype': 'textquery',
                'fields': 'name,formatted_address,international_phone_number,website'
            }
            
            # Appeler l'API
            response = requests.get(
                'https://maps.googleapis.com/maps/api/place/findplacefromtext/json',
                params=params
            )
            
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('candidates'):
                candidate = data['candidates'][0]
                
                place_id = candidate.get('place_id')
                if place_id:
                    # Obtenir plus de détails
                    detail_params = {
                        'key': self.api_keys['google_maps'],
                        'place_id': place_id,
                        'fields': 'name,formatted_address,international_phone_number,website,formatted_phone_number'
                    }
                    
                    detail_response = requests.get(
                        'https://maps.googleapis.com/maps/api/place/details/json',
                        params=detail_params
                    )
                    
                    detail_data = detail_response.json()
                    
                    if detail_data.get('status') == 'OK' and detail_data.get('result'):
                        result = detail_data['result']
                        
                        contact = Contact(
                            company_name=result.get('name'),
                            phone=result.get('international_phone_number') or result.get('formatted_phone_number'),
                            address=result.get('formatted_address'),
                            website=result.get('website'),
                            source="gmaps",
                            confidence=0.8  # Haute confiance pour les données GMaps
                        )
                        
                        return contact
            
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche sur Google Maps: {e}")
            return None
    
    def _find_company_info_from_siret(self, siret):
        """Recherche les informations d'une entreprise à partir du SIRET via APIs officielles"""
        # Si pas de SIRET, on ne peut pas chercher
        if not siret:
            return None
            
        # Vérifier que le SIRET est valide (14 chiffres)
        if not re.match(r'^\d{14}$', siret):
            self.logger.warning(f"SIRET invalide: {siret}")
            return None
            
        try:
            # 1. Essayer l'API Entreprise (gouvernement français)
            response = requests.get(
                f"https://entreprise.data.gouv.fr/api/sirene/v3/etablissements/{siret}",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                etablissement = data.get('etablissement', {})
                
                if etablissement:
                    # Extraire l'adresse
                    adresse = etablissement.get('geo_adresse')
                    
                    # Extraire le nom de l'entreprise
                    nom = etablissement.get('unite_legale', {}).get('denomination')
                    if not nom:
                        nom = etablissement.get('denomination_usuelle')
                        
                    contact = Contact(
                        siret=siret,
                        company_name=nom,
                        address=adresse,
                        source="api_entreprise",
                        confidence=0.9  # Très haute confiance pour les données officielles
                    )
                    
                    return contact
            
            # 2. Si l'API Entreprise échoue, essayer Pappers si configuré
            if self.api_keys.get('pappers') and self.api_keys.get('pappers') != 'YOUR_PAPPERS_API_KEY':
                response = requests.get(
                    f"https://api.pappers.fr/v2/entreprise?api_token={self.api_keys['pappers']}&siret={siret}",
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    contact = Contact(
                        siret=siret,
                        company_name=data.get('nom_entreprise'),
                        address=data.get('siege', {}).get('adresse_ligne_1'),
                        phone=data.get('telephone'),
                        email=data.get('email'),
                        website=data.get('site_internet'),
                        source="pappers",
                        confidence=0.85
                    )
                    
                    return contact
            
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche par SIRET: {e}")
            return None
    
    def find_contacts(self, siret=None, url=None, company_name=None, address=None, contact_pages=None):
        """
        Point d'entrée principal pour trouver des contacts.
        Peut utiliser soit un SIRET, soit une URL, soit un nom d'entreprise.
        """
        contacts = []
        
        # 1. Si on a un SIRET, rechercher les infos officielles
        if siret:
            official_info = self._find_company_info_from_siret(siret)
            if official_info:
                contacts.append(official_info)
                # Récupérer le nom et l'adresse pour les recherches suivantes
                company_name = official_info.company_name or company_name
                address = official_info.address or address
                url = official_info.website or url
        
        # 2. Si on a des pages de contact, extraire les informations
        if contact_pages:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                # Trier les pages par priorité
                sorted_pages = sorted(contact_pages, key=lambda p: p.priority if hasattr(p, 'priority') else 0, reverse=True)
                
                # Limiter aux pages les plus pertinentes
                pages_to_analyze = sorted_pages[:min(5, len(sorted_pages))]
                
                # Analyser les pages en parallèle
                future_to_page = {}
                for page in pages_to_analyze:
                    page_url = page.url if hasattr(page, 'url') else page
                    page_priority = page.priority if hasattr(page, 'priority') else 0.5
                    future = executor.submit(self.extract_contacts_from_page, page_url, page_priority)
                    future_to_page[future] = page_url
                
                for future in concurrent.futures.as_completed(future_to_page):
                    page_url = future_to_page[future]
                    try:
                        contact = future.result()
                        if contact:
                            # Si on a le nom de l'entreprise depuis une source officielle, le privilégier
                            if company_name and not contact.company_name:
                                contact.company_name = company_name
                            contacts.append(contact)
                    except Exception as e:
                        self.logger.error(f"Erreur lors de l'analyse de {page_url}: {e}")
        
        # 3. Si on a une URL mais pas de pages de contact spécifiques
        elif url and not contact_pages:
            contact = self.extract_contacts_from_page(url)
            if contact:
                if company_name and not contact.company_name:
                    contact.company_name = company_name
                contacts.append(contact)
        
        # 4. Si on n'a pas trouvé d'email, essayer Google Maps
        if company_name and not any(c.email for c in contacts if c.email):
            gmaps_contact = self._find_contact_info_from_gmaps(company_name, address)
            if gmaps_contact:
                contacts.append(gmaps_contact)
        
        # 5. Fusion et déduplication des contacts
        final_contact = self._merge_contacts(contacts)
        
        # 6. Si on a toujours peu d'informations, essayer des sources alternatives
        if final_contact and not final_contact.email:
            self._enrich_contact_from_alternative_sources(final_contact)
        
        return final_contact
    
    def _merge_contacts(self, contacts):
        """Fusionne plusieurs contacts en un seul, en privilégiant les informations les plus fiables"""
        if not contacts:
            return None
            
        if len(contacts) == 1:
            return contacts[0]
            
        # Trier par niveau de confiance
        sorted_contacts = sorted(contacts, key=lambda c: c.confidence, reverse=True)
        
        # Prendre le contact avec la plus haute confiance comme base
        final_contact = Contact(
            siret=sorted_contacts[0].siret,
            company_name=sorted_contacts[0].company_name,
            confidence=sorted_contacts[0].confidence
        )
        
        # Variables pour savoir si on a déjà trouvé ces informations
        found_email = False
        found_phone = False
        found_address = False
        found_website = False
        
        # Parcourir tous les contacts par ordre de confiance
        for contact in sorted_contacts:
            # Email (prendre le premier trouvé)
            if contact.email and not found_email:
                final_contact.email = contact.email
                found_email = True
            
            # Téléphone (prendre le premier trouvé)
            if contact.phone and not found_phone:
                final_contact.phone = contact.phone
                found_phone = True
            
            # Adresse (prendre la première trouvée)
            if contact.address and not found_address:
                final_contact.address = contact.address
                found_address = True
            
            # Website (prendre le premier trouvé)
            if contact.website and not found_website:
                final_contact.website = contact.website
                found_website = True
            
            # Formulaire de contact (prendre le premier trouvé)
            if contact.form_url and not final_contact.form_url:
                final_contact.form_url = contact.form_url
            
            # Réseaux sociaux (fusionner tous)
            for platform, url in contact.social_media.items():
                if platform not in final_contact.social_media:
                    final_contact.social_media[platform] = url
        
        return final_contact
    
    def _enrich_contact_from_alternative_sources(self, contact):
        """Enrichit un contact avec des sources alternatives quand les méthodes standards échouent"""
        if not contact or (contact.email and contact.phone):
            return
            
        company_name = contact.company_name
        if not company_name:
            return
            
        # 1. Recherche de l'entreprise sur des annuaires en ligne
        directories = [
            f"https://www.pagesjaunes.fr/recherche/{company_name.replace(' ', '+')}",
            f"https://www.societe.com/cgi-bin/search?champs={company_name.replace(' ', '+')}",
            f"https://www.kompass.com/searchCompanies?text={company_name.replace(' ', '+')}"
        ]
        
        # Ne tester que quelques annuaires pour éviter de faire trop de requêtes
        for directory_url in directories[:2]:  # Limiter à 2 annuaires
            try:
                content, _ = self._fetch_url(directory_url)
                if not content:
                    continue
                    
                soup = BeautifulSoup(content, 'html.parser')
                
                # Rechercher l'entreprise dans les résultats
                company_links = []
                
                # Logique spécifique à chaque annuaire
                if "pagesjaunes.fr" in directory_url:
                    results = soup.find_all('a', class_=lambda c: c and 'denomination-links' in c)
                    for result in results:
                        if company_name.lower() in result.get_text().lower():
                            company_links.append(urljoin(directory_url, result['href']))
                
                elif "societe.com" in directory_url:
                    results = soup.find_all('a', class_=lambda c: c and 'nomination_global' in c)
                    for result in results:
                        if company_name.lower() in result.get_text().lower():
                            company_links.append(urljoin(directory_url, result['href']))
                
                # Analyser les pages de l'entreprise trouvées
                for link in company_links[:1]:  # Prendre uniquement le premier résultat
                    content, _ = self._fetch_url(link)
                    if not content:
                        continue
                        
                    page_soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extraire les emails et téléphones
                    if not contact.email:
                        emails = self._extract_emails(page_soup)
                        if emails:
                            contact.email = self._select_best_email(emails)
                            contact.confidence += 0.1
                    
                    if not contact.phone:
                        phones = self._extract_phones(page_soup)
                        if phones:
                            contact.phone = phones[0]
                            contact.confidence += 0.1
                    
                    # Si on a trouvé les deux, on peut s'arrêter
                    if contact.email and contact.phone:
                        return
            
            except Exception as e:
                self.logger.error(f"Erreur lors de la recherche dans l'annuaire {directory_url}: {e}")
        
        # 2. Si tout échoue et que nous avons un site web, essayer de deviner l'email
        if not contact.email and contact.website:
            domain = urlparse(contact.website).netloc
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Formats d'email courants
            common_emails = [
                f"contact@{domain}",
                f"info@{domain}",
                f"hello@{domain}",
                f"bonjour@{domain}",
                f"support@{domain}"
            ]
            
            # Ajouter ces emails avec une confiance très faible
            contact.potential_emails = common_emails
            contact.confidence -= 0.1  # Réduire la confiance car ce sont des suppositions

# Fonction principale
def find_company_contacts(siret=None, url=None, company_name=None):
    """Interface simplifiée pour trouver les contacts d'une entreprise"""
    finder = ContactFinderPro()
    
    # Si on a une URL, d'abord analyser le sitemap
    if url:
        from scrap import SitemapAnalyzer
        analyzer = SitemapAnalyzer()
        results = analyzer.find_important_pages(url)
        contact_pages = results.get('contact_pages', [])
        
        # Trouver les contacts
        contact = finder.find_contacts(
            siret=siret,
            url=url,
            company_name=company_name,
            contact_pages=contact_pages
        )
        
        return contact
    
    # Sinon, utiliser SIRET ou nom d'entreprise
    else:
        return finder.find_contacts(
            siret=siret,
            company_name=company_name
        )

# Exemple d'utilisation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Exemple avec un SIRET
    # contact = find_company_contacts(siret="35350101400045")
    
    # Exemple avec une URL
    contact = find_company_contacts(url="https://www.groupe-pomona.fr/")
    # contact = find_company_contacts(url="https://www.optimail-solutions.com/")
    
    if contact:
        print("\n=== INFORMATIONS DE CONTACT TROUVÉES ===")
        print(f"Entreprise: {contact.company_name}")
        print(f"Email: {contact.email}")
        print(f"Téléphone: {contact.phone}")
        print(f"Adresse: {contact.address}")
        print(f"Site web: {contact.website}")
        print(f"Formulaire de contact: {contact.form_url}")
        print(f"Réseaux sociaux: {contact.social_media}")
        print(f"Source principale: {contact.source}")
        print(f"Indice de confiance: {contact.confidence:.2f}")
        
        if hasattr(contact, 'potential_emails'):
            print("\nEmails potentiels (non vérifiés):")
            for email in contact.potential_emails:
                print(f"  - {email}")
    else:
        print("Aucun contact trouvé.")