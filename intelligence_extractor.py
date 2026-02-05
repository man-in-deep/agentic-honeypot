"""
intelligence_extractor.py
Extracts UPI IDs, bank accounts, phone numbers, etc.
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
            r'ac\s*(?:no|number|#)?[:\s]*([0-9]{9,18})',
            r'savings?\s*account[:\s]*([0-9]{9,18})',
            r'current\s*account[:\s]*([0-9]{9,18})'
        ]
        
        self.phone_patterns = [
            r'\b\d{10}\b',
            r'\+\d{1,3}[-\s]?\d{10}',
            r'phone[:\s]*(\+\d{1,3}[-\s]?\d{10})',
            r'mobile[:\s]*(\d{10})',
            r'contact[:\s]*(\d{10})',
            r'call\s*(me\s*at)?[:\s]*(\d{10})'
        ]
        
        self.scam_keywords = [
            'urgent', 'immediate', 'verify', 'suspend', 'block',
            'secure', 'password', 'hacked', 'compromised',
            'winner', 'prize', 'lottery', 'reward', 'free',
            'payment', 'transfer', 'fee', 'charge', 'money',
            'account', 'bank', 'upi', 'link', 'click',
            'dear customer', 'attention required', 'important'
        ]
    
    def extract_all(self, text: str) -> Dict:
        """Extract all intelligence from text"""
        return {
            "bankAccounts": self.extract_bank_accounts(text),
            "upiIds": self.extract_upi_ids(text),
            "phishingLinks": self.extract_phishing_links(text),
            "phoneNumbers": self.extract_phone_numbers(text),
            "suspiciousKeywords": self.extract_keywords(text)
        }
    
    def extract_upi_ids(self, text: str) -> List[str]:
        """Extract UPI IDs"""
        upi_ids = []
        
        for pattern in self.upi_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                upi = match.group(1) if match.groups() else match.group(0)
                upi = upi.strip().lower()
                if '@' in upi:
                    if any(domain in upi for domain in ['@ok', '@paytm', '@phonepe', '@gpay', '@upi']):
                        upi_ids.append(upi)
        
        return list(set(upi_ids))
    
    def extract_bank_accounts(self, text: str) -> List[str]:
        """Extract bank account numbers"""
        accounts = []
        
        for pattern in self.account_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                account = match.group(1) if match.groups() else match.group(0)
                clean_acc = re.sub(r'\D', '', account)
                if 9 <= len(clean_acc) <= 18:
                    accounts.append(clean_acc)
        
        return list(set(accounts))
    
    def extract_phishing_links(self, text: str) -> List[str]:
        """Extract suspicious URLs"""
        urls = self.url_extractor.find_urls(text)
        phishing = []
        
        suspicious_domains = [
            'verify-account', 'secure-login', 'bank-update',
            'password-reset', 'confirm-identity', 'claim-reward',
            'free-gift', 'lottery-claim', 'prize-winner'
        ]
        
        for url in urls:
            url_lower = url.lower()
            is_phishing = False
            
            for domain in suspicious_domains:
                if domain in url_lower:
                    is_phishing = True
                    break
            
            # Check for URL shorteners
            shorteners = ['bit.ly', 'tinyurl.com', 'shorturl', 'goo.gl', 'ow.ly']
            for shortener in shorteners:
                if shortener in url_lower:
                    is_phishing = True
                    break
            
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                is_phishing = True
            
            if is_phishing:
                phishing.append(url)
        
        return list(set(phishing))
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers"""
        numbers = []
        
        # Try phonenumbers library
        try:
            for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
                formatted = phonenumbers.format_number(
                    match.number,
                    phonenumbers.PhoneNumberFormat.E164
                )
                numbers.append(formatted)
        except:
            pass
        
        # Regex fallback
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
        """Extract suspicious keywords"""
        found = []
        text_lower = text.lower()
        
        for keyword in self.scam_keywords:
            if keyword in text_lower:
                found.append(keyword)
        
        return list(set(found))

# Global instance
intelligence_extractor = IntelligenceExtractor()