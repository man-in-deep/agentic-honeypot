"""
intelligence_extractor.py - Same as before, works fine
"""

import re
from typing import Dict, List
import phonenumbers
from urlextract import URLExtract

class IntelligenceExtractor:
    """Extracts intelligence from scam messages"""
    
    def __init__(self):
        self.url_extractor = URLExtract()
        
        self.upi_patterns = [
            r'[\w\.-]+@(ok\w+|paytm|phonepe|gpay|googlepay|upi|axl|ybl)',
            r'upi://pay\?[^\s]+',
            r'vpa[:\s]*([\w\.-]+@[\w\.-]+)',
            r'upi\s*(?:id)?[:\s]*([\w\.-]+@[\w\.-]+)'
        ]
        
        self.account_patterns = [
            r'account\s*(?:no|number|#)?[:\s]*([0-9]{9,18})',
            r'acc\s*(?:no|number|#)?[:\s]*([0-9]{9,18})',
            r'ac\s*(?:no|number|#)?[:\s]*([0-9]{9,18})'
        ]
        
        self.phone_patterns = [
            r'\b\d{10}\b',
            r'\+\d{1,3}[-\s]?\d{10}',
            r'phone[:\s]*(\+\d{1,3}[-\s]?\d{10})',
            r'mobile[:\s]*(\d{10})'
        ]
        
        self.scam_keywords = [
            'urgent', 'immediate', 'verify', 'suspend', 'block',
            'bank account', 'upi', 'payment', 'transfer', 'money',
            'won', 'prize', 'lottery', 'free', 'winner',
            'click', 'link', 'http://', 'https://',
            'dear customer', 'attention required'
        ]
    
    def extract_all(self, text: str) -> Dict:
        """Extract all intelligence"""
        return {
            "bankAccounts": self.extract_bank_accounts(text),
            "upiIds": self.extract_upi_ids(text),
            "phishingLinks": self.extract_phishing_links(text),
            "phoneNumbers": self.extract_phone_numbers(text),
            "suspiciousKeywords": self.extract_keywords(text)
        }
    
    def extract_upi_ids(self, text: str) -> List[str]:
        upi_ids = []
        for pattern in self.upi_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                upi = match.group(1) if match.groups() else match.group(0)
                upi = upi.strip().lower()
                if '@' in upi:
                    upi_ids.append(upi)
        return list(set(upi_ids))
    
    def extract_bank_accounts(self, text: str) -> List[str]:
        accounts = []
        for pattern in self.account_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                account = match.group(1) if match.groups() else match.group(0)
                clean = re.sub(r'\D', '', account)
                if 9 <= len(clean) <= 18:
                    accounts.append(clean)
        return list(set(accounts))
    
    def extract_phishing_links(self, text: str) -> List[str]:
        urls = self.url_extractor.find_urls(text)
        phishing = []
        for url in urls:
            url_lower = url.lower()
            # Simple phishing detection
            if any(word in url_lower for word in ['verify', 'secure', 'login', 'bank', 'update', 'reset']):
                phishing.append(url)
        return phishing
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        numbers = []
        try:
            for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
                formatted = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
                numbers.append(formatted)
        except:
            pass
        
        for pattern in self.phone_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                for group in match.groups():
                    if group:
                        clean = re.sub(r'[^\d+]', '', group)
                        if len(clean) >= 10:
                            if clean.startswith(('7', '8', '9')) and len(clean) == 10:
                                clean = '+91' + clean
                            numbers.append(clean)
        return list(set(numbers))
    
    def extract_keywords(self, text: str) -> List[str]:
        found = []
        text_lower = text.lower()
        for keyword in self.scam_keywords:
            if keyword in text_lower:
                found.append(keyword)
        return list(set(found))

# Global instance
intelligence_extractor = IntelligenceExtractor()