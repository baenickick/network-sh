import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.text_preprocessing import TextPreprocessor
from utils.network_analysis import NetworkAnalyzer
from utils.topic_modeling import TopicModeler

# 페이지 설정
st.set_page_config(
    page_title="숙박 후기 텍스트 분석 대시보드",
    page_icon="🏨",
    layout="wide"
)

# 세션 상태 초기화
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()
if 'network_analyzer' not in st.session_state:
    st.session_state.network_analyzer = NetworkAnalyzer()
if 'topic_modeler' not in st.session_state:
    st.session_state.topic_modeler = TopicModeler()

# 메인 제목
st.title("🏨 숙박 후기 텍스트 분석 대시보드")
st.markdown("키워드 네트워크 분석과 토픽 모델링을 통한 인터랙티브 텍스트 분석")
st.markdown("---")

# 사이드바 - 파일 업로드 및 설정
with st.sidebar:
    st.header("📁 1. CSV 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "분석할 텍스트가 있는 CSV 파일 업로드",
        type=['csv'],
        help="Context 컬럼에 분석할 텍스트가 있어야 합니다."
    )
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(st.session_state.data)} 행의 데이터가 로드되었습니다.")
            
            # Context 컬럼 확인
            if 'Context' not in st.session_state.data.columns:
                st.error("❌ 'Context' 컬럼이 없습니다!")
                available_columns = list(st.session_state.data.columns)
                st.write("사용 가능한 컬럼:", available_columns)
                st.stop()
                
            # 데이터 미리보기
            with st.expander("데이터 미리보기"):
                st.dataframe(st.session_state.data.head())
                
        except Exception as e:
            st.error(f"파일 로드 중 오류: {e}")
            st.stop()
    
    st.markdown("---")
    
    # 키워드 필터링 옵션
    st.header("🔍 2. 키워드 필터링 설정")
    
    exclude_keywords = st.text_area(
        "제외할 키워드 (쉼표로 구분)",
        placeholder="호텔, 방, 예시키워드",
        help="네트워크 분석과 토픽 모델링에서 제외할 키워드를 입력하세요"
    )
    
    include_keywords = st.text_area(
        "꼭 포함하고 싶은 키워드 (쉼표로 구분)",
        placeholder="서비스, 청결, 위치",
        help="이 키워드들을 포함한 리뷰만 분석합니다. 비워두면 전체 데이터를 분석합니다."
    )
    
    # 키워드 리스트 변환
    exclude_words = [word.strip() for word in exclude_keywords.split(',') if word.strip()] if exclude_keywords else None
    include_words = [word.strip() for word in include_keywords.split(',') if word.strip()] if include_keywords else None
    
    # 필터링 상태 표시
    if exclude_words:
        st.write("🚫 **제외 키워드:**", ", ".join(exclude_words))
    if include_words:
        st.write("✅ **포함 키워드:**", ", ".join(include_words))
    
    st.markdown("---")
    
    # 분석 파라미터
    st.header("⚙️ 3. 분석 설정")
    n_topics = st.slider("토픽 모델링 주제 수", 3, 10, 5)
    min_frequency = st.slider("네트워크 최소 연결 빈도", 1, 5, 2)

# 메인 콘텐츠
if st.session_state.data is not None:
    
    # 실시간 분석 함수
    def perform_realtime_analysis():
        # 텍스트 전처리
        texts = [st.session_state.preprocessor.clean_text(text) for text in st.session_state.data['Context']]
        texts = [text for text in texts if text.strip()]  # 빈 텍스트 제거
        
        if not texts:
            return None, None, None, None
        
        # 동시출현 행렬 생성
        cooccurrence = st.session_state.preprocessor.get_cooccurrence_matrix(
            texts, exclude_words, include_words
        )
        
        # 네트워크 분석
        network_graph = st.session_state.network_analyzer.create_network(
            cooccurrence, min_frequency
        )
        
        # 토픽 모델링을 위한 텍스트 준비
        filtered_texts = []
        for text in texts:
            keywords = st.session_state.preprocessor.extract_keywords(
                text, exclude_words, include_words
            )
            if keywords:  # 키워드가 있는 경우만
                filtered_texts.append(' '.join(keywords))
        
        # 토픽 모델링
        topic_model = None
        if len(filtered_texts) >= 5:  # 최소 5개 문서 필요
            topic_model = st.session_state.topic_modeler.fit_lda_model(
                filtered_texts, n_topics
            )
        
        return network_graph, filtered_texts, len(texts), topic_model
    
    # 분석 실행
    with st.spinner("실시간 분석 중..."):
        network_graph, filtered_texts, total_texts, topic_model = perform_realtime_analysis()
    
    if network_graph is None:
        st.warning("⚠️ 분석할 수 있는 데이터가 없습니다. 키워드 필터를 조정해보세요.")
        st.stop()
    
    # 분석 결과 요약
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 전체 리뷰", len(st.session_state.data))
    with col2:
        st.metric("📝 분석된 리뷰", total_texts)
    with col3:
        st.metric("🔗 네트워크 키워드", len(network_graph.nodes()) if network_graph else 0)
    with col4:
        st.metric("📊 토픽 수", n_topics if topic_model else 0)
    
    # 결과 표시 탭
    tab1, tab2, tab3 = st.tabs(["🕸️ 네트워크 분석", "📊 토픽 모델링", "📈 분석 결과표"])
    
    with tab1:
        st.subheader("키워드 네트워크 분석")
        st.markdown("키워드 간의 동시출현 관계를 시각화합니다. **키워드를 입력하면 실시간으로 업데이트됩니다!**")
        
        # 네트워크 시각화
        network_fig = st.session_state.network_analyzer.create_interactive_network_plot()
        st.plotly_chart(network_fig, use_container_width=True)
        
        # 중심성 지표 테이블
        centrality_df = st.session_state.network_analyzer.calculate_centrality_metrics()
        if not centrality_df.empty:
            st.subheader("📋 키워드 중심성 지표")
            st.markdown("각 키워드의 네트워크 내 영향력을 나타냅니다.")
            st.dataframe(
                centrality_df.head(10), 
                use_container_width=True,
                column_config={
                    "keyword": "키워드",
                    "degree_centrality": "연결 중심성",
                    "betweenness_centrality": "매개 중심성", 
                    "closeness_centrality": "근접 중심성",
                    "eigenvector_centrality": "고유벡터 중심성"
                }
            )
    
    with tab2:
        st.subheader("토픽 모델링 분석")
        st.markdown("LDA 기법을 통해 숨겨진 주제를 발견합니다. **키워드 필터가 실시간 반영됩니다!**")
        
        if topic_model and len(filtered_texts) >= 5:
            # 토픽 분포 차트
            topic_dist_fig = st.session_state.topic_modeler.create_topic_distribution_plot()
            st.plotly_chart(topic_dist_fig, use_container_width=True)
            
            # 토픽별 키워드 테이블
            topics_df = st.session_state.topic_modeler.get_topics_dataframe(n_words=8)
            if not topics_df.empty:
                st.subheader("📋 토픽별 주요 키워드")
                
                # 토픽 요약
                topic_summary = st.session_state.topic_modeler.get_topic_keywords_summary()
                for topic_name, keywords in topic_summary.items():
                    st.markdown(f"**{topic_name}**: {keywords}")
                
                st.markdown("---")
                
                # 상세 토픽 테이블
                for topic in topics_df['topic'].unique():
                    topic_data = topics_df[topics_df['topic'] == topic].head(8)
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"### {topic}")
                    with col2:
                        st.dataframe(
                            topic_data[['keyword', 'weight']].round(4),
                            hide_index=True,
                            column_config={
                                "keyword": "키워드",
                                "weight": "가중치"
                            }
                        )
        else:
            st.warning("🔍 토픽 모델링을 위한 충분한 데이터가 없습니다. (최소 5개 문서 필요)")
            st.info("키워드 필터를 조정하거나 더 많은 데이터를 업로드해보세요.")
    
    with tab3:
        st.subheader("📊 종합 분석 결과")
        
        # 필터링 정보 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 분석 통계")
            analysis_stats = pd.DataFrame({
                '항목': ['업로드된 리뷰 수', '분석된 리뷰 수', '추출된 키워드 수', '네트워크 노드 수', '설정된 토픽 수'],
                '값': [
                    len(st.session_state.data),
                    total_texts,
                    len(set([word for text in filtered_texts for word in text.split()])) if filtered_texts else 0,
                    len(network_graph.nodes()) if network_graph else 0,
                    n_topics if topic_model else 0
                ]
            })
            st.dataframe(analysis_stats, hide_index=True)
        
        with col2:
            st.markdown("### 🔍 필터 설정")
            filter_info = []
            
            if include_words:
                filter_info.append(f"**포함 키워드**: {', '.join(include_words)}")
            else:
                filter_info.append("**포함 키워드**: 전체 데이터 분석")
                
            if exclude_words:
                filter_info.append(f"**제외 키워드**: {', '.join(exclude_words)}")
            else:
                filter_info.append("**제외 키워드**: 없음")
            
            for info in filter_info:
                st.markdown(info)
        
        # 결과 다운로드
        st.markdown("### 💾 결과 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 네트워크 분석 결과 다운로드
            if not centrality_df.empty:
                csv_network = centrality_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📊 네트워크 분석 결과 다운로드",
                    data=csv_network,
                    file_name=f"network_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # 토픽 모델링 결과 다운로드
            if topic_model:
                topics_df = st.session_state.topic_modeler.get_topics_dataframe()
                if not topics_df.empty:
                    csv_topics = topics_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📈 토픽 모델링 결과 다운로드",
                        data=csv_topics,
                        file_name=f"topic_modeling_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

else:
    # 초기 화면
    st.info("👈 **사이드바에서 CSV 파일을 업로드하여 분석을 시작하세요!**")
    
    # 사용 가이드
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 사용 방법
        1. **CSV 파일 업로드**: Context 컬럼에 분석할 텍스트가 있는 파일
        2. **제외 키워드 입력**: 분석에서 빼고 싶은 키워드들
        3. **포함 키워드 입력**: 이 키워드가 있는 리뷰만 분석 (선택사항)
        4. **실시간 분석 결과**: 키워드 입력 시 즉시 차트 업데이트
        5. **결과 다운로드**: 분석 결과를 CSV로 저장
        """)
    
    with col2:
        st.markdown("""
        ### ✨ 주요 기능
        - 🔄 **실시간 인터랙티브 분석**
        - 🕸️ **키워드 네트워크 시각화**  
        - 📊 **토픽 모델링 (LDA)**
        - 📈 **중심성 지표 분석**
        - 💾 **결과 다운로드**
        - 🎯 **키워드 필터링**
        """)
    
    # 샘플 데이터 안내
    st.markdown("---")
    st.markdown("### 📝 샘플 CSV 파일 형식")
    sample_df = pd.DataFrame({
        'Context': [
            '호텔이 깨끗하고 직원들이 매우 친절했습니다. 조식도 맛있었어요.',
            '위치가 좋고 가격이 합리적이었습니다. 다음에 또 올 것 같아요.',
            '방이 좁았지만 시설은 깔끔했습니다. 서비스가 좋았어요.',
            '뷰가 정말 좋았고 침대가 편안했습니다. 추천합니다!'
        ]
    })
    st.dataframe(sample_df, use_container_width=True)
    
    # 샘플 파일 다운로드
    sample_csv = sample_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 샘플 CSV 파일 다운로드",
        data=sample_csv,
        file_name="sample_hotel_reviews.csv",
        mime="text/csv"
    )

# 푸터
st.markdown("---")
st.markdown("**💡 Tip**: 키워드를 입력/수정할 때마다 분석 결과가 실시간으로 업데이트됩니다!")
st.markdown("Built with ❤️ using Streamlit | 도시 브랜딩 연구를 위한 텍스트 분석 도구")

