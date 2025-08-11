import pandas as pd
import re
from konlpy.tag import Okt
from collections import Counter
import numpy as np

class TextPreprocessor:
    def __init__(self):
        self.okt = Okt()
        
    def clean_text(self, text):
        """기본 텍스트 정리"""
        if pd.isna(text):
            return ""
        
        # 특수문자, 숫자 제거
        text = re.sub(r'[^\w\s]', ' ', str(text))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text, exclude_words=None, include_words=None):
        """키워드 추출 및 필터링"""
        if not text:
            return []
            
        # 형태소 분석
        morphs = self.okt.morphs(text, norm=True, stem=True)
        
        # 불용어 제거
        stopwords = {'이', '그', '저', '것', '수', '있', '하', '되', '같', '들', 
                    '많', '좋', '잘', '매우', '정말', '너무', '아주', '완전', '진짜', '한번', '조금'}
        
        if exclude_words:
            stopwords.update(exclude_words)
        
        # 길이가 2 이상인 키워드만 추출
        keywords = [word for word in morphs 
                   if len(word) >= 2 and word not in stopwords]
        
        # 포함할 키워드가 있으면 해당 키워드를 포함한 문장의 키워드만 추출
        if include_words:
            original_text = str(text).lower()
            if not any(inc.lower() in original_text for inc in include_words):
                return []
        
        return keywords
    
    def get_cooccurrence_matrix(self, texts, exclude_words=None, include_words=None, window_size=5):
        """동시출현 행렬 생성"""
        cooccurrence = {}
        
        for text in texts:
            keywords = self.extract_keywords(text, exclude_words, include_words)
            
            # 윈도우 내 키워드 쌍 추출
            for i, word1 in enumerate(keywords):
                for j in range(max(0, i-window_size), min(len(keywords), i+window_size+1)):
                    if i != j:
                        word2 = keywords[j]
                        pair = tuple(sorted([word1, word2]))
                        cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        
        return cooccurrence

