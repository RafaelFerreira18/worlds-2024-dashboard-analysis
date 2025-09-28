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

# Configuração da página
st.set_page_config(
    page_title="LoL Esports Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhorar a aparência
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
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega e retorna os dados dos jogadores"""
    try:
        df = pd.read_csv('player_statistics_cleaned_final.csv')
        return df
    except FileNotFoundError:
        st.error("Arquivo 'player_statistics_cleaned_final.csv' não encontrado!")
        st.stop()

def preprocess_data(df):
    """Preprocessa os dados para análise"""
    # Criar cópia dos dados
    data = df.copy()
    
    # Tratar valores ausentes (representados como '-')
    columns_to_clean = ['Solo Kills', 'FB Victim', 'Country', 'FlashKeybind']
    for col in columns_to_clean:
        if col in data.columns:
            data[col] = data[col].replace('-', np.nan)
    
    # Converter tipos de dados
    numeric_columns = ['Games', 'Win rate', 'KDA', 'Avg kills', 'Avg deaths', 'Avg assists',
                      'CSPerMin', 'GoldPerMin', 'KP%', 'DamagePercent', 'DPM', 'VSPM',
                      'Avg WPM', 'Avg WCPM', 'Avg VWPM', 'GD@15', 'CSD@15', 'XPD@15',
                      'FB %', 'Penta Kills', 'Solo Kills']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Engenharia de variáveis
    data['Kill_Death_Ratio'] = data['Avg kills'] / data['Avg deaths'].replace(0, 0.1)
    data['Efficiency_Score'] = (data['Avg kills'] + data['Avg assists']) / data['Avg deaths'].replace(0, 0.1)
    data['Economic_Efficiency'] = data['GoldPerMin'] / data['CSPerMin'].replace(0, 1)
    data['Early_Game_Advantage'] = (data['GD@15'] + data['CSD@15'] + data['XPD@15']) / 3
    data['Team_Contribution'] = data['KP%'] * data['DamagePercent']
    data['Vision_Control'] = (data['Avg WPM'] + data['Avg WCPM'] + data['Avg VWPM']) / 3
    
    # Classificar jogadores por performance
    performance_metrics = ['KDA', 'Win rate', 'DamagePercent', 'KP%']
    data['Performance_Score'] = data[performance_metrics].fillna(0).mean(axis=1)
    data['Performance_Tier'] = pd.cut(data['Performance_Score'], 
                                     bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                     labels=['Low', 'Medium', 'High', 'Elite'])
    
    return data

def main():
    """Função principal do dashboard"""
    # Cabeçalho principal
    st.markdown('<h1 class="main-header">🎮 LoL Esports Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Análise Estatística e Científica de Dados de Jogadores Profissionais")
    
    # Carregar dados
    df = load_data()
    data = preprocess_data(df)
    
    # Sidebar para navegação
    st.sidebar.title("📊 Navegação")
    page = st.sidebar.selectbox(
        "Escolha uma seção:",
        ["🏠 Visão Geral", "🔍 Análise Exploratória", "🧹 Preparação dos Dados", 
         "📈 Modelagem Estatística", "🧪 Testes de Hipóteses", "📊 Visualizações Interativas", 
         "💡 Soluções Práticas"]
    )
    
    # Filtros laterais
    st.sidebar.markdown("### 🔧 Filtros")
    
    # Filtro por posição
    positions = ['Todas'] + list(data['Position'].unique())
    selected_position = st.sidebar.selectbox("Posição:", positions)
    
    # Filtro por país
    countries = ['Todos'] + [c for c in data['Country'].unique() if pd.notna(c)]
    selected_country = st.sidebar.selectbox("País:", countries)
    
    # Filtro por win rate
    min_winrate, max_winrate = st.sidebar.slider(
        "Win Rate (%)", 
        float(data['Win rate'].min()), 
        float(data['Win rate'].max()), 
        (float(data['Win rate'].min()), float(data['Win rate'].max())),
        format="%.2f"
    )
    
    # Aplicar filtros
    filtered_data = data.copy()
    if selected_position != 'Todas':
        filtered_data = filtered_data[filtered_data['Position'] == selected_position]
    if selected_country != 'Todos':
        filtered_data = filtered_data[filtered_data['Country'] == selected_country]
    filtered_data = filtered_data[
        (filtered_data['Win rate'] >= min_winrate) & 
        (filtered_data['Win rate'] <= max_winrate)
    ]
    
    # Exibir página selecionada
    if page == "🏠 Visão Geral":
        show_overview(data, filtered_data)
    elif page == "🔍 Análise Exploratória":
        show_exploratory_analysis(data, filtered_data)
    elif page == "🧹 Preparação dos Dados":
        show_data_preparation(df, data)
    elif page == "📈 Modelagem Estatística":
        show_statistical_modeling(data, filtered_data)
    elif page == "🧪 Testes de Hipóteses":
        show_hypothesis_testing(data, filtered_data)
    elif page == "📊 Visualizações Interativas":
        show_interactive_visualizations(data, filtered_data)
    elif page == "💡 Soluções Práticas":
        show_practical_solutions(data, filtered_data)

def show_overview(data, filtered_data):
    """Mostra visão geral do dataset"""
    st.markdown('<h2 class="section-header">🏠 Visão Geral do Dataset</h2>', unsafe_allow_html=True)
    
    # Informações básicas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Jogadores", len(data))
    with col2:
        st.metric("Times Únicos", data['TeamName'].nunique())
    with col3:
        st.metric("Países Representados", data['Country'].nunique())
    with col4:
        st.metric("Posições", data['Position'].nunique())
    
    # Contexto do dataset
    st.markdown("""
    <div class="insight-box">
    <h3>📋 Contexto do Dataset</h3>
    <p>Este dataset contém estatísticas de jogadores profissionais de League of Legends, 
    incluindo métricas de performance individual, econômica e de equipe. Os dados permitem 
    análises sobre fatores que contribuem para o sucesso em partidas competitivas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estatísticas descritivas básicas
    st.markdown("### 📈 Estatísticas Descritivas")
    
    key_metrics = ['Win rate', 'KDA', 'Avg kills', 'Avg deaths', 'DamagePercent', 'GoldPerMin']
    stats_df = filtered_data[key_metrics].describe().round(3)
    st.dataframe(stats_df, use_container_width=True)
    
    # Distribuição por posição
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Distribuição por Posição")
        position_counts = filtered_data['Position'].value_counts()
        fig = px.pie(values=position_counts.values, names=position_counts.index,
                    title="Distribuição de Jogadores por Posição")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Top 10 Países")
        country_counts = filtered_data['Country'].value_counts().head(10)
        fig = px.bar(x=country_counts.values, y=country_counts.index,
                    orientation='h', title="Jogadores por País")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def show_exploratory_analysis(data, filtered_data):
    """Análise exploratória detalhada"""
    st.markdown('<h2 class="section-header">🔍 Análise Exploratória de Dados</h2>', unsafe_allow_html=True)
    
    # Identificação de outliers
    st.markdown("### 🎯 Identificação de Outliers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Boxplot para identificar outliers
        metric_for_outliers = st.selectbox(
            "Selecione uma métrica para análise de outliers:",
            ['KDA', 'Win rate', 'DamagePercent', 'GoldPerMin', 'Avg kills']
        )
        
        fig = px.box(filtered_data, y=metric_for_outliers, x='Position',
                    title=f"Distribuição de {metric_for_outliers} por Posição")
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas dos outliers
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
        # Top performers
        st.markdown("### 🏆 Top Performers")
        top_performers = filtered_data.nlargest(5, 'Performance_Score')[
            ['PlayerName', 'TeamName', 'Position', 'Performance_Score', 'Win rate', 'KDA']
        ].round(3)
        st.dataframe(top_performers, use_container_width=True)
        
        # Piores performers
        st.markdown("### 📉 Jogadores com Menor Performance")
        bottom_performers = filtered_data.nsmallest(5, 'Performance_Score')[
            ['PlayerName', 'TeamName', 'Position', 'Performance_Score', 'Win rate', 'KDA']
        ].round(3)
        st.dataframe(bottom_performers, use_container_width=True)
    
    # Matriz de correlação
    st.markdown("### 📊 Matriz de Correlação")
    
    correlation_metrics = ['Win rate', 'KDA', 'Avg kills', 'DamagePercent', 'GoldPerMin', 
                          'KP%', 'CSPerMin', 'Kill_Death_Ratio', 'Performance_Score']
    
    corr_matrix = filtered_data[correlation_metrics].corr()
    
    fig = px.imshow(corr_matrix, 
                    title="Matriz de Correlação entre Métricas de Performance",
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights da correlação
    st.markdown("""
    <div class="insight-box">
    <h4>🔍 Insights da Correlação:</h4>
    <ul>
    <li><strong>KDA e Win Rate:</strong> Correlação forte positiva - jogadores com melhor KDA tendem a vencer mais</li>
    <li><strong>Damage Percent e KP%:</strong> Jogadores que causam mais dano participam mais dos kills da equipe</li>
    <li><strong>Gold Per Min e CS Per Min:</strong> Eficiência econômica ligada à capacidade de farmar</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise por posição
    st.markdown("### 🎭 Análise por Posição")
    
    position_stats = filtered_data.groupby('Position')[
        ['Win rate', 'KDA', 'DamagePercent', 'GoldPerMin', 'KP%']
    ].mean().round(3)
    
    st.dataframe(position_stats, use_container_width=True)
    
    # Gráfico de radar por posição
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
            
            # Normalizar valores para 0-1 para melhor visualização
            normalized_values = []
            for i, metric in enumerate(metrics_radar):
                min_val = filtered_data[metric].min()
                max_val = filtered_data[metric].max()
                norm_val = (values[i] - min_val) / (max_val - min_val)
                normalized_values.append(norm_val)
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],  # Fechar o polígono
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
    """Mostra o processo de preparação dos dados"""
    st.markdown('<h2 class="section-header">🧹 Preparação e Limpeza dos Dados</h2>', unsafe_allow_html=True)
    
    # Comparação antes e depois
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dados Originais")
        st.write(f"**Shape:** {original_data.shape}")
        st.write(f"**Colunas:** {original_data.shape[1]}")
        
        # Valores ausentes originais
        missing_original = original_data.isnull().sum()
        missing_original = missing_original[missing_original > 0]
        if len(missing_original) > 0:
            st.write("**Valores ausentes:**")
            st.dataframe(missing_original, use_container_width=True)
        else:
            st.write("**Valores ausentes:** Nenhum")
    
    with col2:
        st.markdown("### ✨ Dados Processados")
        st.write(f"**Shape:** {processed_data.shape}")
        st.write(f"**Colunas:** {processed_data.shape[1]}")
        
        # Valores ausentes processados
        missing_processed = processed_data.isnull().sum()
        missing_processed = missing_processed[missing_processed > 0]
        if len(missing_processed) > 0:
            st.write("**Valores ausentes restantes:**")
            st.dataframe(missing_processed, use_container_width=True)
        else:
            st.write("**Valores ausentes:** Todos tratados")
    
    # Transformações aplicadas
    st.markdown("### 🔧 Engenharia de Variáveis")
    
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
    
    # Distribuições das novas variáveis
    st.markdown("### 📊 Distribuição das Novas Variáveis")
    
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
    
    # Estatísticas de qualidade dos dados
    st.markdown("### ✅ Qualidade dos Dados")
    
    quality_metrics = {
        'Completude': f"{(processed_data.notna().sum().sum() / processed_data.size) * 100:.2f}%",
        'Consistência': "Tipos de dados corrigidos e padronizados",
        'Outliers': f"{len(processed_data)} registros analisados para outliers",
        'Duplicatas': f"{processed_data.duplicated().sum()} duplicatas encontradas"
    }
    
    quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Métrica', 'Status'])
    st.dataframe(quality_df, use_container_width=True)

def show_statistical_modeling(data, filtered_data):
    """Modelagem estatística com regressão linear"""
    st.markdown('<h2 class="section-header">📈 Modelagem Estatística</h2>', unsafe_allow_html=True)
    
    # Seleção de variáveis para o modelo
    st.markdown("### 🎯 Configuração do Modelo")
    
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
    
    # Preparar dados para modelagem
    model_data = filtered_data[selected_features + [target_var]].dropna()
    
    if len(model_data) < 10:
        st.error("Dados insuficientes para modelagem. Ajuste os filtros.")
        return
    
    X = model_data[selected_features]
    y = model_data[target_var]
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predições
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas do modelo
    st.markdown("### 📊 Performance do Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Treino", f"{r2_score(y_train, y_pred_train):.3f}")
    with col2:
        st.metric("R² Teste", f"{r2_score(y_test, y_pred_test):.3f}")
    with col3:
        st.metric("RMSE Treino", f"{np.sqrt(mean_squared_error(y_train, y_pred_train)):.3f}")
    with col4:
        st.metric("RMSE Teste", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
    
    # Coeficientes do modelo
    st.markdown("### 📋 Coeficientes do Modelo")
    
    coef_df = pd.DataFrame({
        'Variável': selected_features,
        'Coeficiente': model.coef_,
        'Importância': np.abs(model.coef_)
    }).sort_values('Importância', ascending=False)
    
    st.dataframe(coef_df, use_container_width=True)
    
    # Gráfico de importância das variáveis
    fig = px.bar(coef_df, x='Importância', y='Variável', orientation='h',
                title="Importância das Variáveis (Valor Absoluto dos Coeficientes)")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise de resíduos
    st.markdown("### 🔍 Análise de Resíduos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de valores preditos vs reais
        fig = px.scatter(x=y_test, y=y_pred_test, 
                        title="Valores Preditos vs Reais",
                        labels={'x': 'Valores Reais', 'y': 'Valores Preditos'})
        
        # Linha de referência y=x
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', name='Linha Ideal', 
                                line=dict(dash='dash', color='red')))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribuição dos resíduos
        residuals = y_test - y_pred_test
        fig = px.histogram(x=residuals, title="Distribuição dos Resíduos",
                          labels={'x': 'Resíduos'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Modelo estatístico detalhado com statsmodels
    st.markdown("### 📈 Análise Estatística Detalhada")
    
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    
    # Mostrar resumo estatístico
    st.text(str(model_sm.summary()))
    
    # Teste de heterocedasticidade
    _, pvalue_bp, _, _ = het_breuschpagan(model_sm.resid, X_sm)
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>🧪 Diagnóstico do Modelo:</h4>
    <ul>
    <li><strong>R²:</strong> {model_sm.rsquared:.3f} - Explica {model_sm.rsquared*100:.1f}% da variância</li>
    <li><strong>R² Ajustado:</strong> {model_sm.rsquared_adj:.3f}</li>
    <li><strong>F-statistic:</strong> {model_sm.fvalue:.2f} (p-value: {model_sm.f_pvalue:.4f})</li>
    <li><strong>Teste Breusch-Pagan:</strong> p-value = {pvalue_bp:.4f} {'(Homocedasticidade)' if pvalue_bp > 0.05 else '(Heterocedasticidade detectada)'}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Predições para novos dados
    st.markdown("### 🎯 Fazer Predições")
    
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

if __name__ == "__main__":
    main()
