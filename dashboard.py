import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="LoL Esports Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.25rem;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .insight-box h3, .insight-box h4 {
        color: #1f77b4;
        margin-top: 0;
    }
    .insight-box p, .insight-box ul, .insight-box li {
        color: #2c3e50;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
        margin: 1rem 0;
        color: #856404;
    }
    .warning-box h3, .warning-box h4 {
        color: #b8860b;
        margin-top: 0;
    }
    .warning-box p, .warning-box ul, .warning-box li {
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('player_statistics_cleaned_final.csv')
        return df
    except FileNotFoundError:
        st.error("Arquivo 'player_statistics_cleaned_final.csv' não encontrado!")
        st.stop()

def preprocess_data(df):
    data = df.copy()
    
    columns_to_clean = ['Solo Kills', 'FB Victim', 'Country', 'FlashKeybind']
    for col in columns_to_clean:
        if col in data.columns:
            data[col] = data[col].replace('-', np.nan)
    
    numeric_columns = ['Games', 'Win rate', 'KDA', 'Avg kills', 'Avg deaths', 'Avg assists',
                      'CSPerMin', 'GoldPerMin', 'KP%', 'DamagePercent', 'DPM', 'VSPM',
                      'Avg WPM', 'Avg WCPM', 'Avg VWPM', 'GD@15', 'CSD@15', 'XPD@15',
                      'FB %', 'Penta Kills', 'Solo Kills']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data['Kill_Death_Ratio'] = data['Avg kills'] / data['Avg deaths'].replace(0, 0.1)
    data['Efficiency_Score'] = (data['Avg kills'] + data['Avg assists']) / data['Avg deaths'].replace(0, 0.1)
    data['Economic_Efficiency'] = data['GoldPerMin'] / data['CSPerMin'].replace(0, 1)
    data['Early_Game_Advantage'] = (data['GD@15'] + data['CSD@15'] + data['XPD@15']) / 3
    data['Team_Contribution'] = data['KP%'] * data['DamagePercent']
    data['Vision_Control'] = (data['Avg WPM'] + data['Avg WCPM'] + data['Avg VWPM']) / 3
    
    performance_metrics = ['KDA', 'Win rate', 'DamagePercent', 'KP%']
    data['Performance_Score'] = data[performance_metrics].fillna(0).mean(axis=1)
    data['Performance_Tier'] = pd.cut(data['Performance_Score'], 
                                     bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                     labels=['Low', 'Medium', 'High', 'Elite'])
    
    return data

def main():
    st.markdown('<h1 class="main-header">LoL Esports Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Análise Estatística e Científica de Dados de Jogadores Profissionais")
    
    df = load_data()
    data = preprocess_data(df)
    
    st.sidebar.title("Navegação")
    page = st.sidebar.selectbox(
        "Escolha uma seção:",
        ["Visão Geral", "Análise Exploratória", "Preparação dos Dados", 
         "Modelagem Estatística", "Testes de Hipóteses", "Visualizações Interativas", 
         "Soluções Práticas"]
    )
    
    st.sidebar.markdown("### Filtros")
    
    positions = ['Todas'] + list(data['Position'].unique())
    selected_position = st.sidebar.selectbox("Posição:", positions)
    
    countries = ['Todos'] + [c for c in data['Country'].unique() if pd.notna(c)]
    selected_country = st.sidebar.selectbox("País:", countries)
    
    min_winrate, max_winrate = st.sidebar.slider(
        "Win Rate (%)", 
        float(data['Win rate'].min()), 
        float(data['Win rate'].max()), 
        (float(data['Win rate'].min()), float(data['Win rate'].max())),
        format="%.2f"
    )
    
    filtered_data = data.copy()
    if selected_position != 'Todas':
        filtered_data = filtered_data[filtered_data['Position'] == selected_position]
    if selected_country != 'Todos':
        filtered_data = filtered_data[filtered_data['Country'] == selected_country]
    filtered_data = filtered_data[
        (filtered_data['Win rate'] >= min_winrate) & 
        (filtered_data['Win rate'] <= max_winrate)
    ]
    
    if page == "Visão Geral":
        show_overview(data, filtered_data)
    elif page == "Análise Exploratória":
        show_exploratory_analysis(data, filtered_data)
    elif page == "Preparação dos Dados":
        show_data_preparation(df, data)
    elif page == "Modelagem Estatística":
        show_statistical_modeling(data, filtered_data)
    elif page == "Testes de Hipóteses":
        show_hypothesis_testing(data, filtered_data)
    elif page == "Visualizações Interativas":
        show_interactive_visualizations(data, filtered_data)
    elif page == "Soluções Práticas":
        show_practical_solutions(data, filtered_data)

def show_overview(data, filtered_data):
    st.markdown('<h2 class="section-header">Visão Geral do Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Jogadores", len(data))
    with col2:
        st.metric("Times Únicos", data['TeamName'].nunique())
    with col3:
        st.metric("Países Representados", data['Country'].nunique())
    with col4:
        st.metric("Posições", data['Position'].nunique())
    
    st.markdown("""
    <div class="insight-box">
    <h3>Contexto do Dataset</h3>
    <p>Este dataset contém estatísticas de jogadores profissionais de League of Legends, 
    incluindo métricas de performance individual, econômica e de equipe. Os dados permitem 
    análises sobre fatores que contribuem para o sucesso em partidas competitivas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Estatísticas Descritivas")
    
    key_metrics = ['Win rate', 'KDA', 'Avg kills', 'Avg deaths', 'DamagePercent', 'GoldPerMin']
    stats_df = filtered_data[key_metrics].describe().round(3)
    st.dataframe(stats_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribuição por Posição")
        position_counts = filtered_data['Position'].value_counts()
        fig = px.pie(values=position_counts.values, names=position_counts.index,
                    title="Distribuição de Jogadores por Posição")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Top 10 Países")
        country_counts = filtered_data['Country'].value_counts().head(10)
        fig = px.bar(x=country_counts.values, y=country_counts.index,
                    orientation='h', title="Jogadores por País")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def show_exploratory_analysis(data, filtered_data):
    st.markdown('<h2 class="section-header">Análise Exploratória de Dados</h2>', unsafe_allow_html=True)
    
    st.markdown("### Identificação de Outliers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric_for_outliers = st.selectbox(
            "Selecione uma métrica para análise de outliers:",
            ['KDA', 'Win rate', 'DamagePercent', 'GoldPerMin', 'Avg kills']
        )
        
        fig = px.box(filtered_data, y=metric_for_outliers, x='Position',
                    title=f"Distribuição de {metric_for_outliers} por Posição")
        st.plotly_chart(fig, use_container_width=True)
        
        Q1 = filtered_data[metric_for_outliers].quantile(0.25)
        Q3 = filtered_data[metric_for_outliers].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = filtered_data[
            (filtered_data[metric_for_outliers] < lower_bound) | 
            (filtered_data[metric_for_outliers] > upper_bound)
        ]
        st.metric("Outliers Identificados", len(outliers))
    
    with col2:
        st.markdown("### Top Performers")
        top_performers = filtered_data.nlargest(5, 'Performance_Score')[
            ['PlayerName', 'TeamName', 'Position', 'Performance_Score', 'Win rate', 'KDA']
        ].round(3)
        st.dataframe(top_performers, use_container_width=True)
        
        st.markdown("### Jogadores com Menor Performance")
        bottom_performers = filtered_data.nsmallest(5, 'Performance_Score')[
            ['PlayerName', 'TeamName', 'Position', 'Performance_Score', 'Win rate', 'KDA']
        ].round(3)
        st.dataframe(bottom_performers, use_container_width=True)
    
    st.markdown("### Matriz de Correlação")
    
    correlation_metrics = ['Win rate', 'KDA', 'Avg kills', 'DamagePercent', 'GoldPerMin', 
                          'KP%', 'CSPerMin', 'Kill_Death_Ratio', 'Performance_Score']
    
    corr_matrix = filtered_data[correlation_metrics].corr()
    
    fig = px.imshow(corr_matrix, 
                    title="Matriz de Correlação entre Métricas de Performance",
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>Insights da Correlação:</h4>
    <ul>
    <li><strong>KDA e Win Rate:</strong> Correlação forte positiva - jogadores com melhor KDA tendem a vencer mais</li>
    <li><strong>Damage Percent e KP%:</strong> Jogadores que causam mais dano participam mais dos kills da equipe</li>
    <li><strong>Gold Per Min e CS Per Min:</strong> Eficiência econômica ligada à capacidade de farmar</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Análise por Posição")
    
    position_stats = filtered_data.groupby('Position')[
        ['Win rate', 'KDA', 'DamagePercent', 'GoldPerMin', 'KP%']
    ].mean().round(3)
    
    st.dataframe(position_stats, use_container_width=True)
    
    positions = filtered_data['Position'].unique()
    selected_positions = st.multiselect(
        "Selecione posições para comparação:",
        positions,
        default=positions[:3] if len(positions) >= 3 else positions
    )
    
    if selected_positions:
        fig = go.Figure()
        
        metrics_radar = ['Win rate', 'KDA', 'DamagePercent', 'KP%', 'GoldPerMin']
        
        for position in selected_positions:
            pos_data = filtered_data[filtered_data['Position'] == position]
            values = [pos_data[metric].mean() for metric in metrics_radar]
            
            normalized_values = []
            for i, metric in enumerate(metrics_radar):
                min_val = filtered_data[metric].min()
                max_val = filtered_data[metric].max()
                norm_val = (values[i] - min_val) / (max_val - min_val)
                normalized_values.append(norm_val)
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],
                theta=metrics_radar + [metrics_radar[0]],
                fill='toself',
                name=position
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparação de Performance por Posição (Normalizado)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_data_preparation(original_data, processed_data):
    st.markdown('<h2 class="section-header">Preparação e Limpeza dos Dados</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dados Originais")
        st.write(f"**Shape:** {original_data.shape}")
        st.write(f"**Colunas:** {original_data.shape[1]}")
        
        missing_original = original_data.isnull().sum()
        missing_original = missing_original[missing_original > 0]
        if len(missing_original) > 0:
            st.write("**Valores ausentes:**")
            st.dataframe(missing_original, use_container_width=True)
        else:
            st.write("**Valores ausentes:** Nenhum")
    
    with col2:
        st.markdown("### Dados Processados")
        st.write(f"**Shape:** {processed_data.shape}")
        st.write(f"**Colunas:** {processed_data.shape[1]}")
        
        missing_processed = processed_data.isnull().sum()
        missing_processed = missing_processed[missing_processed > 0]
        if len(missing_processed) > 0:
            st.write("**Valores ausentes restantes:**")
            st.dataframe(missing_processed, use_container_width=True)
        else:
            st.write("**Valores ausentes:** Todos tratados")
    
    st.markdown("### Engenharia de Variáveis")
    
    new_features = {
        'Kill_Death_Ratio': 'Avg kills / Avg deaths (tratando divisão por zero)',
        'Efficiency_Score': '(Avg kills + Avg assists) / Avg deaths',
        'Economic_Efficiency': 'GoldPerMin / CSPerMin',
        'Early_Game_Advantage': 'Média de (GD@15 + CSD@15 + XPD@15) / 3',
        'Team_Contribution': 'KP% × DamagePercent',
        'Vision_Control': 'Média de (Avg WPM + Avg WCPM + Avg VWPM) / 3',
        'Performance_Score': 'Média normalizada de KDA, Win rate, DamagePercent, KP%',
        'Performance_Tier': 'Categorização em Low, Medium, High, Elite'
    }
    
    features_df = pd.DataFrame(list(new_features.items()), columns=['Variável', 'Descrição'])
    st.dataframe(features_df, use_container_width=True)
    
    st.markdown("### Distribuição das Novas Variáveis")
    
    new_vars = ['Kill_Death_Ratio', 'Efficiency_Score', 'Performance_Score']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=new_vars,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, var in enumerate(new_vars, 1):
        fig.add_trace(
            go.Histogram(x=processed_data[var], name=var, showlegend=False),
            row=1, col=i
        )
    
    fig.update_layout(title_text="Distribuições das Variáveis Criadas")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Qualidade dos Dados")
    
    quality_metrics = {
        'Completude': f"{(processed_data.notna().sum().sum() / processed_data.size) * 100:.2f}%",
        'Consistência': "Tipos de dados corrigidos e padronizados",
        'Outliers': f"{len(processed_data)} registros analisados para outliers",
        'Duplicatas': f"{processed_data.duplicated().sum()} duplicatas encontradas"
    }
    
    quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Métrica', 'Status'])
    st.dataframe(quality_df, use_container_width=True)

def show_statistical_modeling(data, filtered_data):
    st.markdown('<h2 class="section-header">Modelagem Estatística</h2>', unsafe_allow_html=True)
    
    st.markdown("### Configuração do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_var = st.selectbox(
            "Variável Dependente (Target):",
            ['Win rate', 'Performance_Score', 'KDA', 'DamagePercent']
        )
    
    with col2:
        available_features = ['Avg kills', 'Avg deaths', 'Avg assists', 'GoldPerMin', 
                             'CSPerMin', 'KP%', 'DamagePercent', 'Kill_Death_Ratio', 
                             'Efficiency_Score', 'Team_Contribution']
        
        selected_features = st.multiselect(
            "Variáveis Independentes:",
            available_features,
            default=['Kill_Death_Ratio', 'GoldPerMin', 'KP%', 'DamagePercent']
        )
    
    if not selected_features:
        st.warning("Selecione pelo menos uma variável independente.")
        return
    
    model_data = filtered_data[selected_features + [target_var]].dropna()
    
    if len(model_data) < 10:
        st.error("Dados insuficientes para modelagem. Ajuste os filtros.")
        return
    
    X = model_data[selected_features]
    y = model_data[target_var]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    st.markdown("### Performance do Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Treino", f"{r2_score(y_train, y_pred_train):.3f}")
    with col2:
        st.metric("R² Teste", f"{r2_score(y_test, y_pred_test):.3f}")
    with col3:
        st.metric("RMSE Treino", f"{np.sqrt(mean_squared_error(y_train, y_pred_train)):.3f}")
    with col4:
        st.metric("RMSE Teste", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
    
    st.markdown("### Coeficientes do Modelo")
    
    coef_df = pd.DataFrame({
        'Variável': selected_features,
        'Coeficiente': model.coef_,
        'Importância': np.abs(model.coef_)
    }).sort_values('Importância', ascending=False)
    
    st.dataframe(coef_df, use_container_width=True)
    
    fig = px.bar(coef_df, x='Importância', y='Variável', orientation='h',
                title="Importância das Variáveis (Valor Absoluto dos Coeficientes)")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Análise de Resíduos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(x=y_test, y=y_pred_test, 
                        title="Valores Preditos vs Reais",
                        labels={'x': 'Valores Reais', 'y': 'Valores Preditos'})
        
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', name='Linha Ideal', 
                                line=dict(dash='dash', color='red')))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        residuals = y_test - y_pred_test
        fig = px.histogram(x=residuals, title="Distribuição dos Resíduos",
                          labels={'x': 'Resíduos'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Análise Estatística Detalhada")
    
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    
    st.text(str(model_sm.summary()))
    
    _, pvalue_bp, _, _ = het_breuschpagan(model_sm.resid, X_sm)
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>Diagnóstico do Modelo:</h4>
    <ul>
    <li><strong>R²:</strong> {model_sm.rsquared:.3f} - Explica {model_sm.rsquared*100:.1f}% da variância</li>
    <li><strong>R² Ajustado:</strong> {model_sm.rsquared_adj:.3f}</li>
    <li><strong>F-statistic:</strong> {model_sm.fvalue:.2f} (p-value: {model_sm.f_pvalue:.4f})</li>
    <li><strong>Teste Breusch-Pagan:</strong> p-value = {pvalue_bp:.4f} {'(Homocedasticidade)' if pvalue_bp > 0.05 else '(Heterocedasticidade detectada)'}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Fazer Predições")
    
    with st.expander("Predizer Performance"):
        pred_values = {}
        for feature in selected_features:
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            default_val = float(X[feature].mean())
            
            pred_values[feature] = st.slider(
                f"{feature}:",
                min_val, max_val, default_val
            )
        
        if st.button("Fazer Predição"):
            pred_input = np.array(list(pred_values.values())).reshape(1, -1)
            prediction = model.predict(pred_input)[0]
            
            st.success(f"**Predição para {target_var}:** {prediction:.3f}")

def show_hypothesis_testing(data, filtered_data):
    st.markdown('<h2 class="section-header">Testes de Hipóteses</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Esta seção aplica testes estatísticos para validar insights obtidos na análise exploratória,
    utilizando intervalos de confiança e testes de significância.
    """)
    
    st.markdown("### Teste 1: Diferença de Performance entre Posições")
    
    positions = list(filtered_data['Position'].unique())
    col1, col2 = st.columns(2)
    
    with col1:
        pos1 = st.selectbox("Primeira posição:", positions, index=0)
    with col2:
        pos2 = st.selectbox("Segunda posição:", positions, index=1 if len(positions) > 1 else 0)
    
    if pos1 != pos2:
        data_pos1 = filtered_data[filtered_data['Position'] == pos1]['Performance_Score'].dropna()
        data_pos2 = filtered_data[filtered_data['Position'] == pos2]['Performance_Score'].dropna()
        
        if len(data_pos1) > 1 and len(data_pos2) > 1:
            t_stat, p_value = stats.ttest_ind(data_pos1, data_pos2)
            
            conf_interval_pos1 = stats.t.interval(0.95, len(data_pos1)-1, 
                                                 loc=data_pos1.mean(), 
                                                 scale=stats.sem(data_pos1))
            conf_interval_pos2 = stats.t.interval(0.95, len(data_pos2)-1, 
                                                 loc=data_pos2.mean(), 
                                                 scale=stats.sem(data_pos2))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Média {pos1}", f"{data_pos1.mean():.3f}")
                st.caption(f"IC 95%: [{conf_interval_pos1[0]:.3f}, {conf_interval_pos1[1]:.3f}]")
            with col2:
                st.metric(f"Média {pos2}", f"{data_pos2.mean():.3f}")
                st.caption(f"IC 95%: [{conf_interval_pos2[0]:.3f}, {conf_interval_pos2[1]:.3f}]")
            with col3:
                st.metric("p-value", f"{p_value:.4f}")
                st.caption("Significativo" if p_value < 0.05 else "Não significativo")
            
            if p_value < 0.05:
                st.markdown(f"""
                <div class="insight-box">
                <h4>Resultado Significativo</h4>
                <p>Há diferença estatisticamente significativa entre a performance de jogadores 
                {pos1} e {pos2} (p = {p_value:.4f} < 0.05).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                <h4>Resultado Não Significativo</h4>
                <p>Não há diferença estatisticamente significativa entre a performance de jogadores 
                {pos1} e {pos2} (p = {p_value:.4f} ≥ 0.05).</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("### Teste 2: Correlação entre KDA e Win Rate")
    
    common_data = filtered_data[['KDA', 'Win rate']].dropna()
    
    if len(common_data) > 2:
        corr_coef, corr_p_value = stats.pearsonr(common_data['KDA'], common_data['Win rate'])
        
        spear_coef, spear_p_value = stats.spearmanr(common_data['KDA'], common_data['Win rate'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pearson r", f"{corr_coef:.3f}")
        with col2:
            st.metric("p-value", f"{corr_p_value:.4f}")
        with col3:
            st.metric("Spearman ρ", f"{spear_coef:.3f}")
        with col4:
            st.metric("p-value", f"{spear_p_value:.4f}")
        
        fig = px.scatter(common_data, x='KDA', y='Win rate', 
                        title="Correlação entre KDA e Win Rate",
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        
        if abs(corr_coef) < 0.3:
            strength = "fraca"
        elif abs(corr_coef) < 0.7:
            strength = "moderada"
        else:
            strength = "forte"
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>Interpretação da Correlação</h4>
        <ul>
        <li><strong>Correlação Pearson:</strong> {corr_coef:.3f} (correlação {strength})</li>
        <li><strong>Significância:</strong> {'Significativa' if corr_p_value < 0.05 else 'Não significativa'} (p = {corr_p_value:.4f})</li>
        <li><strong>Interpretação:</strong> {'Existe relação linear significativa' if corr_p_value < 0.05 else 'Não há relação linear significativa'} entre KDA e Win Rate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_interactive_visualizations(data, filtered_data):
    st.markdown('<h2 class="section-header">Visualizações Interativas</h2>', unsafe_allow_html=True)
    
    st.markdown("### Explorador de Relações Multi-dimensional")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x_var = st.selectbox("Eixo X:", 
                           ['KDA', 'Win rate', 'DamagePercent', 'GoldPerMin', 'Performance_Score'])
    with col2:
        y_var = st.selectbox("Eixo Y:", 
                           ['Win rate', 'KDA', 'DamagePercent', 'GoldPerMin', 'Performance_Score'])
    with col3:
        size_var = st.selectbox("Tamanho:", 
                              ['Games', 'KP%', 'CSPerMin', 'Avg kills'])
    with col4:
        color_var = st.selectbox("Cor:", 
                               ['Position', 'Performance_Tier', 'Country'])
    
    fig = px.scatter(
        filtered_data, 
        x=x_var, 
        y=y_var,
        size=size_var,
        color=color_var,
        hover_data=['PlayerName', 'TeamName', 'Position', 'Win rate', 'KDA'],
        title=f"{y_var} vs {x_var} (tamanho: {size_var}, cor: {color_var})",
        width=800,
        height=600
    )
    
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Mapa de Calor: Performance por Time")
    
    team_stats = filtered_data.groupby('TeamName').agg({
        'Win rate': 'mean',
        'KDA': 'mean',
        'DamagePercent': 'mean',
        'GoldPerMin': 'mean',
        'Performance_Score': 'mean'
    }).round(3)
    
    team_stats_normalized = (team_stats - team_stats.min()) / (team_stats.max() - team_stats.min())
    
    fig = px.imshow(
        team_stats_normalized.T,
        x=team_stats_normalized.index,
        y=team_stats_normalized.columns,
        color_continuous_scale='RdYlBu_r',
        title="Performance Normalizada por Time",
        aspect='auto'
    )
    
    fig.update_layout(
        xaxis_title="Times",
        yaxis_title="Métricas",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_practical_solutions(data, filtered_data):
    st.markdown('<h2 class="section-header">Soluções Práticas e Recomendações</h2>', unsafe_allow_html=True)
    
    st.markdown("### Recomendações Estratégicas")
    
    top_performers = filtered_data.nlargest(10, 'Performance_Score')
    
    top_stats = top_performers[['KDA', 'Win rate', 'DamagePercent', 'KP%', 'GoldPerMin', 'CSPerMin']].mean()
    overall_stats = filtered_data[['KDA', 'Win rate', 'DamagePercent', 'KP%', 'GoldPerMin', 'CSPerMin']].mean()
    
    st.markdown("#### Características dos Top Performers")
    
    comparison_df = pd.DataFrame({
        'Métrica': top_stats.index,
        'Top 10 Jogadores': top_stats.values,
        'Média Geral': overall_stats.values,
        'Diferença (%)': ((top_stats.values - overall_stats.values) / overall_stats.values * 100)
    }).round(3)
    
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>Insights para Melhoria de Performance:</h4>
    <ul>
    <li><strong>KDA Superior:</strong> Top performers mantêm KDA {((top_stats['KDA'] - overall_stats['KDA']) / overall_stats['KDA'] * 100):.1f}% mais alto que a média</li>
    <li><strong>Participação em Kills:</strong> Maior KP% indica melhor coordenação de equipe</li>
    <li><strong>Eficiência Econômica:</strong> Melhor farm (CS/min) e conversão em ouro</li>
    <li><strong>Impacto no Dano:</strong> Maior % de dano da equipe</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Recomendações Específicas por Posição")
    
    position_recommendations = {
        'Top': ['Controle de Lane e Teamfight', 'Focar no farm early game', 'Melhorar teleports'],
        'Jungle': ['Map Control e Ganks', 'Maximizar presença no mapa', 'Coordenar objetivos'],
        'Mid': ['Damage e Roaming', 'Balancear farm com fights', 'Melhorar wave management'],
        'Adc': ['DPS e Posicionamento', 'Focar em positioning', 'Melhorar farm'],
        'Support': ['Vision e Utility', 'Maximizar vision control', 'Melhorar roaming']
    }
    
    selected_pos = st.selectbox("Selecione uma posição:", list(position_recommendations.keys()))
    
    if selected_pos:
        recommendations = position_recommendations[selected_pos]
        st.markdown(f"#### Recomendações para {selected_pos}:")
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    st.markdown("### Limitações do Estudo")
    
    st.markdown("""
    <div class="warning-box">
    <h4>Limitações Importantes:</h4>
    <ul>
    <li><strong>Dados Temporais:</strong> Análise baseada em snapshot</li>
    <li><strong>Contexto de Patches:</strong> Mudanças no jogo podem afetar métricas</li>
    <li><strong>Meta Game:</strong> Estratégias podem influenciar performance</li>
    <li><strong>Fatores Externos:</strong> Coaching e ambiente não são considerados</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
