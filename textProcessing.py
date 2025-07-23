import spacy
import re
import difflib
from typing import Dict, List


class EnhancedBusinessCardParser:
    def __init__(self):
        # Load spaCy English model
        self.nlp = spacy.load("en_core_web_sm")

        # Regex patterns
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'(\+?91[-\s]?\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'
        )

        # Exclusion terms
        self.address_terms = {
            'sector', 'floor', 'street', 'road', 'avenue', 'gzb', 'delhi', 'mumbai',
            'bangalore', 'chennai', 'pune', 'hyderabad', 'vaishali', 'cloud'
        }
        self.business_terms = {
            'security', 'services', 'executive', 'field', 'manager', 'director',
            'company', 'ltd', 'pvt', 'corp', 'inc', 'llc', 'officer', 'cliff'
        }
        self.building_indicators = {
            'tower', 'plaza', 'complex', 'center', 'mall', 'building', 'cloud'
        }

        # Common Indian surname suffixes
        self.indian_suffixes = ['singh', 'kumar', 'sharma', 'gupta', 'yadav', 'patel']

    def extract_info(self, text_list: List[str]) -> Dict:
        """
        Extract a single name, emails, and phone numbers.
        """
        print(text_list)  # Debug

        full_text = " ".join(text_list)

        # Extract emails and phones
        emails = self.email_pattern.findall(full_text)
        phones = self.phone_pattern.findall(full_text)

        # Extract person name
        primary_name = self._extract_primary_name(text_list, full_text)

        return {
            'name': primary_name,
            'emails': emails,
            'phones': phones
        }

    def _extract_primary_name(self, text_list: List[str], full_text: str) -> str:
        """
        Find the single best candidate for a person name.
        """
        candidates = []

        # Strategy 1: spaCy NER (PERSON entities)
        doc = self.nlp(full_text.title())
        for ent in doc.ents:
            if ent.label_ == "PERSON" and self._is_valid_name(ent.text):
                candidates.append(ent.text.title())

        # Strategy 2: Pattern-based fallback
        for text in text_list:
            clean_text = re.sub(r'[^A-Za-z ]', '', text).strip()
            if self._is_valid_name(clean_text):
                candidates.append(clean_text.title())

        # Deduplicate
        candidates = list(set(candidates))

        if not candidates:
            return None  # No name found

        # Rank candidates
        scored = [(self._name_score(name), name) for name in candidates]
        scored.sort(reverse=True)  # Highest score first
        return scored[0][1]  # Return best name

    def _is_valid_name(self, name: str) -> bool:
        """
        Apply filtering rules to eliminate false positives.
        """
        name_lower = name.lower()
        words = name_lower.split()

        # Reject if too short or too long
        if not (2 <= len(words) <= 3):
            return False

        # Reject if contains numbers or special chars
        if re.search(r'[\d\-+()]', name):
            return False

        # Reject if contains address or building terms
        if any(term in name_lower for term in (self.address_terms | self.building_indicators)):
            return False

        # Fuzzy match to business terms
        for word in words:
            if difflib.get_close_matches(word, self.business_terms, n=1, cutoff=0.8):
                return False

        return True

    def _name_score(self, name: str) -> int:
        """
        Score a name based on likelihood of being a person name.
        """
        score = 0
        words = name.split()

        # Favor 2-word names slightly more than 3-word names
        if len(words) == 2:
            score += 50
        elif len(words) == 3:
            score += 40

        # Favor title-cased names
        if all(word.istitle() for word in words):
            score += 20

        # Boost for Indian surnames
        if any(suffix in name.lower() for suffix in self.indian_suffixes):
            score += 30

        # Penalize for uncommon/strange words
        if any(word.lower() in self.business_terms for word in words):
            score -= 30

        return score
