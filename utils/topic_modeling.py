from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

class TopicModeler:
    def __init__(self):
        self.lda_model = None
        self.vectorizer = None
        self.doc_topic_matrix = None
        
    def fit_lda_model(self, texts, n_topics=5, max_features=100):
        """LDA 토픽 모델링"""
        if not texts or all(not text.strip() for text in texts):
            return None
            
        # 벡터화
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            token_pattern=r'\b\w{2,}\b'
        )
        
        try:
            doc_term_matrix = self.vectorizer.fit_transform(texts)
            
            if doc_term_matrix.sum() == 0:
                return None
                
            # LDA 모델 훈련
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            
            self.doc_topic_matrix = self.lda_model.fit_transform(doc_term_matrix)
            return self.lda_model
            
        except Exception as e:
            print(f"LDA 모델 훈련 오류: {e}")
            return None
    
    def get_topics_dataframe(self, n_words=10):
        """토픽별 주요 키워드 DataFrame 반환"""
        if not self.lda_model or not self.vectorizer:
            return pd.DataFrame()
            
        feature_names = self.vectorizer.get_feature_names_out()
        topics_data = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            for word, weight in zip(top_words, top_weights):
                topics_data.append({
                    'topic': f'토픽 {topic_idx + 1}',
                    'keyword': word,
                    'weight': round(weight, 4)
                })
        
        return pd.DataFrame(topics_data)
    
    def create_topic_distribution_plot(self):
        """토픽 분포 시각화"""
        if self.doc_topic_matrix is None:
            return go.Figure().add_annotation(
                text="토픽 모델이 훈련되지 않았습니다.", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # 토픽별 평균 확률
        topic_probs = self.doc_topic_matrix.mean(axis=0)
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f'토픽 {i+1}' for i in range(len(topic_probs))],
                y=topic_probs,
                marker_color='lightblue',
                text=[f'{prob:.3f}' for prob in topic_probs],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='토픽별 문서 분포',
            xaxis_title='토픽',
            yaxis_title='평균 확률',
            plot_bgcolor='white'
        )
        
        return fig
    
    def get_topic_keywords_summary(self):
        """토픽별 키워드 요약"""
        if not self.lda_model or not self.vectorizer:
            return {}
            
        feature_names = self.vectorizer.get_feature_names_out()
        topic_summary = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-5:][::-1]  # 상위 5개만
            top_words = [feature_names[i] for i in top_words_idx]
            topic_summary[f'토픽 {topic_idx + 1}'] = ', '.join(top_words)
            
        return topic_summary

