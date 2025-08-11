import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import pandas as pd

class NetworkAnalyzer:
    def __init__(self):
        self.graph = None
        
    def create_network(self, cooccurrence_dict, min_frequency=2):
        """네트워크 그래프 생성"""
        self.graph = nx.Graph()
        
        # 빈도수가 임계값 이상인 쌍만 추가
        for (word1, word2), freq in cooccurrence_dict.items():
            if freq >= min_frequency:
                self.graph.add_edge(word1, word2, weight=freq)
        
        return self.graph
    
    def calculate_centrality_metrics(self):
        """중심성 지표 계산"""
        if not self.graph or len(self.graph.nodes()) == 0:
            return pd.DataFrame()
            
        # 다양한 중심성 지표 계산
        degree_cent = nx.degree_centrality(self.graph)
        betweenness_cent = nx.betweenness_centrality(self.graph)
        closeness_cent = nx.closeness_centrality(self.graph)
        
        try:
            eigenvector_cent = nx.eigenvector_centrality(self.graph, max_iter=1000)
        except:
            eigenvector_cent = {node: 0 for node in self.graph.nodes()}
        
        # DataFrame으로 정리
        centrality_df = pd.DataFrame({
            'keyword': list(degree_cent.keys()),
            'degree_centrality': list(degree_cent.values()),
            'betweenness_centrality': list(betweenness_cent.values()),
            'closeness_centrality': list(closeness_cent.values()),
            'eigenvector_centrality': list(eigenvector_cent.values())
        }).round(4)
        
        return centrality_df.sort_values('degree_centrality', ascending=False)
    
    def create_interactive_network_plot(self):
        """인터랙티브 네트워크 시각화"""
        if not self.graph or len(self.graph.nodes()) == 0:
            return go.Figure().add_annotation(
                text="분석할 데이터가 없습니다.<br>키워드 필터를 조정해보세요.", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        
        # 레이아웃 계산
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # 노드 정보
        node_trace = go.Scatter(
            x=[pos[node][0] for node in self.graph.nodes()],
            y=[pos[node][1] for node in self.graph.nodes()],
            mode='markers+text',
            text=list(self.graph.nodes()),
            textposition="middle center",
            textfont=dict(size=12, color="white"),
            marker=dict(
                size=[self.graph.degree(node) * 5 + 15 for node in self.graph.nodes()],
                color=[self.graph.degree(node) for node in self.graph.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="연결도"),
                line=dict(width=2, color="white")
            ),
            hovertemplate='<b>%{text}</b><br>연결도: %{marker.color}<extra></extra>',
            name='키워드'
        )
        
        # 엣지 정보
        edge_x, edge_y = [], []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            name='연결'
        )
        
        # 그래프 생성
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title="키워드 네트워크 분석",
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig

