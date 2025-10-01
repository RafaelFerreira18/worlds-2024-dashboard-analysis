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
            ['KDA', 'DamagePercent', 'GoldPerMin', 'Avg kills']
        )
        
        fig = px.box(filtered_data, y=metric_for_outliers, x='Position',
                    title=f"Distribuição de {metric_for_outliers} por Posição")
        st.plotly_chart(fig, use_container_width=True)
        
        Q1 = filtered_data[metric_for_outliers].quantile(0.25)
        Q3 = filtered_data[metric_for_outliers].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
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
                          'KP%', 'CSPerMin', 'Performance_Score']
    
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
        ['KDA', 'DamagePercent', 'GoldPerMin', 'KP%']
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
        
        metrics_radar = ['KDA', 'DamagePercent', 'KP%', 'GoldPerMin']
        
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

def show_statistical_modeling(data, filtered_data):
    st.markdown('<h2 class="section-header">Modelagem Estatística</h2>', unsafe_allow_html=True)
    
    st.markdown("### Configuração do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_var = st.selectbox(
            "Variável Dependente (Target):",
            ['Performance_Score', 'Win rate', 'KDA', 'DamagePercent']
        )
    
    with col2:
        available_features = ['Avg kills', 'Avg deaths', 'Avg assists', 'GoldPerMin', 
                     'CSPerMin', 'KP%', 'DamagePercent', 'DPM', 'VSPM',
                     'Avg WPM', 'Avg WCPM', 'Avg VWPM', 'GD@15', 'CSD@15', 'XPD@15',
                     'FB %', 'Penta Kills', 'Solo Kills', 'KDA', 
                     'Efficiency_Score', 'Team_Contribution', 'Economic_Efficiency',
                     'Early_Game_Advantage', 'Vision_Control', 'Games']
        
        selected_features = st.multiselect(
            "Variáveis Independentes:",
            available_features,
            default=['KDA', 'GoldPerMin', 'DamagePercent']
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

    tab1, tab2, tab3 = st.tabs(["Resumo do Modelo", "Coeficientes Detalhados", "Diagnósticos"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="R² (Ajustado)",
                value=f"{model_sm.rsquared_adj:.3f}",
                delta=f"{(model_sm.rsquared_adj - model_sm.rsquared):.3f}",
                help="R² ajustado penaliza modelos com muitas variáveis"
            )
        
        with col2:
            st.metric(
                label="F-statistic",
                value=f"{model_sm.fvalue:.2f}",
                help="Testa se o modelo é melhor que a média simples"
            )
        
        with col3:
            significance = "Significativo" if model_sm.f_pvalue < 0.05 else "Não significativo"
            st.metric(
                label="P-value (F-test)",
                value=f"{model_sm.f_pvalue:.4f}",
                delta=significance,
                help="Se p < 0.05, o modelo é estatisticamente válido"
            )
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>Interpretação:</h4>
        <p>O modelo explica <strong>{model_sm.rsquared*100:.1f}%</strong> da variância em {target_var}. 
        Com R² ajustado de <strong>{model_sm.rsquared_adj:.3f}</strong>, isso significa que as variáveis 
        selecionadas têm poder preditivo {'forte' if model_sm.rsquared_adj > 0.7 else 'moderado' if model_sm.rsquared_adj > 0.4 else 'fraco'}.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        coef_summary = pd.DataFrame({
            'Variável': model_sm.params.index,
            'Coeficiente': model_sm.params.values,
            'Erro Padrão': model_sm.bse.values,
            'p-value': model_sm.pvalues.values,
            'IC 95% Min': model_sm.conf_int()[0].values,
            'IC 95% Max': model_sm.conf_int()[1].values
        }).round(4)
        
        coef_summary = coef_summary[coef_summary['Variável'] != 'const']
        
        coef_summary['Significativo'] = coef_summary['p-value'].apply(
            lambda x: 'Sim' if x < 0.05 else 'Não'
        )
        
        st.dataframe(
            coef_summary.style.background_gradient(subset=['Coeficiente'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        st.markdown("""
        <div class="insight-box">
        <h4>Como interpretar:</h4>
        <ul>
        <li><strong>Coeficiente positivo:</strong> Aumentar a variável aumenta o target</li>
        <li><strong>Coeficiente negativo:</strong> Aumentar a variável diminui o target</li>
        <li><strong>p-value < 0.05:</strong> A variável é estatisticamente significativa</li>
        <li><strong>Intervalo de Confiança:</strong> Range onde o verdadeiro coeficiente provavelmente está</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            _, pvalue_bp, _, _ = het_breuschpagan(model_sm.resid, X_sm)
            
            st.markdown("#### Teste de Breusch-Pagan")
            st.markdown(f"**P-value:** {pvalue_bp:.4f}")
            
            if pvalue_bp > 0.05:
                st.success("Homocedasticidade confirmada (variância constante)")
            else:
                st.warning("Heterocedasticidade detectada (variância não constante)")
        
        with col2:
            from scipy.stats import shapiro
            
            if len(model_sm.resid) <= 5000:
                stat_shapiro, p_shapiro = shapiro(model_sm.resid)
                
                st.markdown("#### Teste de Normalidade (Shapiro-Wilk)")
                st.markdown(f"**P-value:** {p_shapiro:.4f}")
                
                if p_shapiro > 0.05:
                    st.success("Resíduos seguem distribuição normal")
                else:
                    st.warning("Resíduos não são normais (pode afetar intervalos de confiança)")
            else:
                st.info("Dataset muito grande para teste de Shapiro-Wilk. Use Q-Q plot para avaliar normalidade.")
        
        st.markdown("#### Estatísticas Adicionais")
        
    
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
            
            if target_var == 'Win rate':
                min_val, max_val = 0, 1
                unit = "%"
                multiplier = 100
            elif target_var == 'Performance_Score':
                min_val, max_val = 0, 1
                unit = ""
                multiplier = 1
            elif target_var == 'KDA':
                min_val, max_val = 0, 10
                unit = ""
                multiplier = 1
            elif target_var == 'DamagePercent':
                min_val, max_val = 0, 1
                unit = "%"
                multiplier = 100
            
            prediction_clipped = np.clip(prediction, min_val, max_val)
            
            if prediction < min_val or prediction > max_val:
                st.warning(f"""
                ⚠️ **Extrapolação Detectada:** O modelo previu **{prediction:.3f}**, mas esse valor 
                está fora do range válido para {target_var} ({min_val}-{max_val}).
                
                A predição foi ajustada para **{prediction_clipped:.3f}**.
                
                **Por que isso acontece?** Os valores de entrada estão muito diferentes dos dados 
                que o modelo viu durante o treinamento. Reduza ou aumente os valores nos sliders.
                """)

            if unit == "%":
                st.success(f"**Predição para {target_var}:** {prediction_clipped*multiplier:.1f}%")
            else:
                st.success(f"**Predição para {target_var}:** {prediction_clipped:.3f}")

def show_hypothesis_testing(data, filtered_data):
    st.markdown('<h2 class="section-header">Testes de Hipóteses</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Esta seção aplica testes t para validar insights obtidos na análise exploratória,
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
    
    st.markdown("### Teste 2: KDA Alto vs KDA Baixo - Impacto no Win Rate")
    
    common_data = filtered_data[['KDA', 'Win rate']].dropna()
    
    if len(common_data) > 10:
        kda_median = common_data['KDA'].median()
        
        high_kda_group = common_data[common_data['KDA'] > kda_median]['Win rate']
        low_kda_group = common_data[common_data['KDA'] <= kda_median]['Win rate']
        
        if len(high_kda_group) > 1 and len(low_kda_group) > 1:
            t_stat, p_value = stats.ttest_ind(high_kda_group, low_kda_group)
            
            conf_interval_high = stats.t.interval(0.95, len(high_kda_group)-1, 
                                                 loc=high_kda_group.mean(), 
                                                 scale=stats.sem(high_kda_group))
            conf_interval_low = stats.t.interval(0.95, len(low_kda_group)-1, 
                                                loc=low_kda_group.mean(), 
                                                scale=stats.sem(low_kda_group))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Win Rate - KDA Alto", f"{high_kda_group.mean():.3f}")
                st.caption(f"IC 95%: [{conf_interval_high[0]:.3f}, {conf_interval_high[1]:.3f}]")
                st.caption(f"n = {len(high_kda_group)}")
            with col2:
                st.metric(f"Win Rate - KDA Baixo", f"{low_kda_group.mean():.3f}")
                st.caption(f"IC 95%: [{conf_interval_low[0]:.3f}, {conf_interval_low[1]:.3f}]")
                st.caption(f"n = {len(low_kda_group)}")
            with col3:
                st.metric("p-value", f"{p_value:.4f}")
                st.caption("Significativo" if p_value < 0.05 else "Não significativo")
                st.metric("Diferença", f"{(high_kda_group.mean() - low_kda_group.mean()):.3f}")
            
            fig = px.box(
                x=['KDA Alto' if kda > kda_median else 'KDA Baixo' for kda in common_data['KDA']],
                y=common_data['Win rate'],
                title=f"Distribuição do Win Rate por Grupo de KDA (mediana = {kda_median:.2f})"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if p_value < 0.05:
                st.markdown(f"""
                <div class="insight-box">
                <h4>Resultado Significativo</h4>
                <p>Jogadores com KDA alto têm Win Rate significativamente {'maior' if high_kda_group.mean() > low_kda_group.mean() else 'menor'} 
                que jogadores com KDA baixo (p = {p_value:.4f} < 0.05).</p>
                <p><strong>Diferença média:</strong> {abs(high_kda_group.mean() - low_kda_group.mean()):.3f} pontos no Win Rate</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                <h4>Resultado Não Significativo</h4>
                <p>Não há diferença estatisticamente significativa no Win Rate entre jogadores 
                com KDA alto e baixo (p = {p_value:.4f} ≥ 0.05).</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("### Teste 3: Impacto do Gold Per Minute no Damage Percent")
    
    gpm_data = filtered_data[['GoldPerMin', 'DamagePercent']].dropna()
    
    if len(gpm_data) > 10:
        gpm_median = gpm_data['GoldPerMin'].median()
        
        high_gpm_group = gpm_data[gpm_data['GoldPerMin'] > gpm_median]['DamagePercent']
        low_gpm_group = gpm_data[gpm_data['GoldPerMin'] <= gpm_median]['DamagePercent']
        
        if len(high_gpm_group) > 1 and len(low_gpm_group) > 1:
            t_stat, p_value = stats.ttest_ind(high_gpm_group, low_gpm_group)
            
            conf_interval_high = stats.t.interval(0.95, len(high_gpm_group)-1, 
                                                 loc=high_gpm_group.mean(), 
                                                 scale=stats.sem(high_gpm_group))
            conf_interval_low = stats.t.interval(0.95, len(low_gpm_group)-1, 
                                                loc=low_gpm_group.mean(), 
                                                scale=stats.sem(low_gpm_group))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Damage% - GPM Alto", f"{high_gpm_group.mean():.3f}")
                st.caption(f"IC 95%: [{conf_interval_high[0]:.3f}, {conf_interval_high[1]:.3f}]")
                st.caption(f"n = {len(high_gpm_group)}")
            with col2:
                st.metric(f"Damage% - GPM Baixo", f"{low_gpm_group.mean():.3f}")
                st.caption(f"IC 95%: [{conf_interval_low[0]:.3f}, {conf_interval_low[1]:.3f}]")
                st.caption(f"n = {len(low_gpm_group)}")
            with col3:
                st.metric("p-value", f"{p_value:.4f}")
                st.caption("Significativo" if p_value < 0.05 else "Não significativo")
                st.metric("Diferença", f"{(high_gpm_group.mean() - low_gpm_group.mean()):.3f}")
            
            if p_value < 0.05:
                st.markdown(f"""
                <div class="insight-box">
                <h4>Resultado Significativo</h4>
                <p>Jogadores com Gold Per Minute alto têm Damage Percent significativamente {'maior' if high_gpm_group.mean() > low_gpm_group.mean() else 'menor'} 
                que jogadores com GPM baixo (p = {p_value:.4f} < 0.05).</p>
                <p><strong>Interpretação:</strong> Maior eficiência econômica está associada a maior contribuição de dano.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                <h4>Resultado Não Significativo</h4>
                <p>Não há diferença estatisticamente significativa no Damage Percent entre jogadores 
                com GPM alto e baixo (p = {p_value:.4f} ≥ 0.05).</p>
                </div>
                """, unsafe_allow_html=True)

def show_interactive_visualizations(data, filtered_data):
    st.markdown('<h2 class="section-header">Visualizações Interativas</h2>', unsafe_allow_html=True)
    
    st.markdown("### Explorador de Relações Multi-dimensional")

    viz_type = st.radio("Tipo de visualização:", ["Scatter Plot", "Top Performers", "Comparação por Grupo"])

    if viz_type == "Scatter Plot":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("Eixo X:", 
                            ['KDA', 'Win rate', 'DamagePercent', 'GoldPerMin', 'Performance_Score'])
        with col2:
            y_var = st.selectbox("Eixo Y:", 
                            ['Win rate', 'KDA', 'DamagePercent', 'GoldPerMin', 'Performance_Score'])
        with col3:
            color_var = st.selectbox("Cor:", 
                                ['Position', 'Performance_Tier'])
        
        fig = px.scatter(
            filtered_data, 
            x=x_var, 
            y=y_var,
            color=color_var,
            hover_data=['PlayerName', 'TeamName', 'Win rate', 'KDA'],
            title=f"{y_var} vs {x_var}",
            opacity=0.7
        )
        
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Top Performers":
        metric = st.selectbox("Métrica para ranking:", 
                            ['Performance_Score', 'Win rate', 'KDA', 'DamagePercent'])
        top_n = st.slider("Mostrar top:", 5, 20, 10)
        
        top_data = filtered_data.nlargest(top_n, metric)[['PlayerName', 'TeamName', 'Position', metric]]
        
        fig = px.bar(
            top_data, 
            x=metric, 
            y='PlayerName',
            color='Position',
            orientation='h',
            title=f"Top {top_n} Jogadores por {metric}",
            hover_data=['TeamName']
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    else: 
        group_by = st.selectbox("Agrupar por:", ['Position', 'Performance_Tier'])
        metrics = st.multiselect("Métricas:", 
                                ['Win rate', 'KDA', 'DamagePercent', 'GoldPerMin'],
                                default=['Win rate', 'KDA'])
        
        grouped = filtered_data.groupby(group_by)[metrics].mean().reset_index()
        
        fig = px.bar(
            grouped, 
            x=group_by, 
            y=metrics,
            barmode='group',
            title=f"Comparação de Métricas por {group_by}"
        )
        st.plotly_chart(fig, use_container_width=True)

    
    st.markdown("### Mapa de Calor: Performance por Time")
    team_counts = filtered_data['TeamName'].value_counts()
    valid_teams = team_counts[team_counts >= 2].index
    filtered_teams = filtered_data[filtered_data['TeamName'].isin(valid_teams)]

    if len(valid_teams) < 2:
        st.warning("Dados insuficientes. Ajuste os filtros para incluir mais times.")
    else:
        team_stats = filtered_teams.groupby('TeamName').agg({
            'Win rate': 'mean',
            'KDA': 'mean',
            'DamagePercent': 'mean',
            'GoldPerMin': 'mean',
            'Performance_Score': 'mean'
        }).round(3)
        
        team_stats_normalized = team_stats.copy()
        for col in team_stats.columns:
            min_val = team_stats[col].min()
            max_val = team_stats[col].max()
            
            if max_val - min_val > 0:
                team_stats_normalized[col] = (team_stats[col] - min_val) / (max_val - min_val)
            else:
                team_stats_normalized[col] = 0.5 
        
        if len(team_stats_normalized) > 15:
            team_stats_normalized = team_stats_normalized.nlargest(15, 'Performance_Score')
        
        fig = px.imshow(
            team_stats_normalized.T,
            x=team_stats_normalized.index,
            y=team_stats_normalized.columns,
            color_continuous_scale='RdYlGn',
            title="Performance Normalizada por Time (0 = pior, 1 = melhor)",
            aspect='auto',
            labels=dict(color="Score Normalizado")
        )
        
        fig.update_layout(
            xaxis_title="Times",
            yaxis_title="Métricas",
            height=500,
            xaxis={'tickangle': -45}
        )
        
        fig.update_traces(text=team_stats_normalized.T.round(2), texttemplate='%{text}')
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver valores originais (não normalizados)"):
            st.dataframe(team_stats.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

def show_practical_solutions(data, filtered_data):
    st.markdown('<h2 class="section-header">Soluções Práticas e Recomendações</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Esta seção traduz os insights estatísticos obtidos nos testes de hipóteses em 
    recomendações práticas e acionáveis para melhoria de performance dos jogadores.
    """)
    
    st.markdown("### Recomendações Baseadas em Evidências Estatísticas")
    
    st.markdown("#### 1. Correlação KDA-Win Rate: Foco na Sobrevivência")
    
    common_data = filtered_data[['KDA', 'Win rate']].dropna()
    if len(common_data) > 2:
        corr_coef, corr_p_value = stats.pearsonr(common_data['KDA'], common_data['Win rate'])
        
        if corr_p_value < 0.05:
            st.markdown(f"""
            <div class="insight-box">
            <h4>Evidência Estatística Confirmada</h4>
            <p><strong>Correlação KDA-Win Rate:</strong> {corr_coef:.3f} (p = {corr_p_value:.4f})</p>
            <p><strong>Recomendação Prática:</strong> O teste confirma que melhorar o KDA tem impacto 
            direto no Win Rate. Priorize estratégias de sobrevivência e participação em kills.</p>
            </div>
            """, unsafe_allow_html=True)
            
            kda_quartiles = filtered_data['KDA'].quantile([0.25, 0.5, 0.75])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                low_kda = filtered_data[filtered_data['KDA'] <= kda_quartiles[0.25]]['Win rate'].mean()
                st.metric("Win Rate - KDA Baixo (Q1)", f"{low_kda:.3f}")
                
            with col2:
                mid_kda = filtered_data[
                    (filtered_data['KDA'] > kda_quartiles[0.25]) & 
                    (filtered_data['KDA'] <= kda_quartiles[0.75])
                ]['Win rate'].mean()
                st.metric("Win Rate - KDA Médio (Q2-Q3)", f"{mid_kda:.3f}")
                
            with col3:
                high_kda = filtered_data[filtered_data['KDA'] > kda_quartiles[0.75]]['Win rate'].mean()
                st.metric("Win Rate - KDA Alto (Q4)", f"{high_kda:.3f}")
        
        else:
            st.markdown(f"""
            <div class="warning-box">
            <h4>Correlação Não Significativa</h4>
            <p>A correlação entre KDA e Win Rate não é estatisticamente significativa 
            (p = {corr_p_value:.4f}). Outras variáveis podem ser mais importantes.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("#### 2. Diferenças entre Posições: Estratégias Personalizadas")

    positions = list(filtered_data['Position'].unique())

    if len(positions) >= 2:
        st.markdown("**Análise de Significância entre Posições:**")
        
        position_stats = filtered_data.groupby('Position')['Performance_Score'].agg(['mean', 'std', 'count']).round(3)
        position_stats = position_stats.sort_values('mean', ascending=False)
        
        st.dataframe(position_stats, use_container_width=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>Ranking de Performance por Posição</h4>
        <p><strong>Melhor Performance:</strong> {position_stats.index[0]} (Score: {position_stats.iloc[0]['mean']:.3f})</p>
        <p><strong>Maior Oportunidade:</strong> {position_stats.index[-1]} (Score: {position_stats.iloc[-1]['mean']:.3f})</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected_pos = st.selectbox(
            "Selecione uma posição para análise detalhada:", 
            positions,
            help="Cada posição tem métricas-chave específicas"
        )
        
        if selected_pos:
            position_metrics = {
                'Top': {
                    'key_metrics': ['CSPerMin', 'GD@15', 'CSD@15', 'KP%', 'Solo Kills'],
                    'description': 'Controle de Lane e Impacto em Teamfights',
                    'priorities': [
                        'Maximizar farm early game para vantagem de itens',
                        'Melhorar gestão de wave para evitar ganks',
                        'Participar de teamfights com TPs estratégicos',
                        'Desenvolver profundidade de champion pool'
                    ]
                },
                'Jungle': {
                    'key_metrics': ['Early_Game_Advantage', 'Team_Contribution', 'KP%', 'VSPM', 'GD@15'],
                    'description': 'Map Control e Presença Global',
                    'priorities': [
                        'Otimizar pathing e clear speed',
                        'Maximizar presença em objetivos (Dragon/Herald)',
                        'Coordenar ganks com timing de power spikes',
                        'Melhorar vision control e deep warding'
                    ]
                },
                'Mid': {
                    'key_metrics': ['DamagePercent', 'KP%', 'CSPerMin', 'CSD@15', 'GoldPerMin'],
                    'description': 'Damage Output e Roaming',
                    'priorities': [
                        'Balancear farm com participação em skirmishes',
                        'Melhorar wave management para roams',
                        'Maximizar damage em teamfights',
                        'Coordenar com jungle para vision e ganks'
                    ]
                },
                'Adc': {
                    'key_metrics': ['DamagePercent', 'DPM', 'CSPerMin', 'GoldPerMin', 'KDA'],
                    'description': 'DPS Sustentado e Positioning',
                    'priorities': [
                        'Focar em positioning para maximizar uptime de DPS',
                        'Melhorar CS e eficiência econômica',
                        'Coordenar com support para lane dominance',
                        'Desenvolver decision making em teamfights'
                    ]
                },
                'Support': {
                    'key_metrics': ['KP%', 'Avg assists', 'VSPM', 'Avg WPM', 'Avg WCPM', 'Vision_Control'],
                    'description': 'Vision Control e Utility',
                    'priorities': [
                        'Maximizar vision score e controle de mapa',
                        'Melhorar roaming e map presence',
                        'Coordenar engages e disengages',
                        'Otimizar uso de wards e sweeper'
                    ]
                }
            }
            
            pos_key = selected_pos
            if selected_pos not in position_metrics:
                for key in position_metrics.keys():
                    if key.lower().startswith(selected_pos.lower()[:3]):
                        pos_key = key
                        break
            
            if pos_key in position_metrics:
                pos_info = position_metrics[pos_key]
                
                st.markdown(f"""
                <div class="insight-box">
                <h4>{pos_key}: {pos_info['description']}</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**Métricas-Chave para {pos_key}:**")
                
                pos_data = filtered_data[filtered_data['Position'] == selected_pos]
                
                available_metrics = [m for m in pos_info['key_metrics'] if m in pos_data.columns]
                
                if available_metrics:
                    metrics_analysis = []
                    for metric in available_metrics:
                        metric_data = pos_data[metric].dropna()
                        if len(metric_data) > 0:
                            q25 = metric_data.quantile(0.25)
                            q50 = metric_data.quantile(0.50)
                            q75 = metric_data.quantile(0.75)
                            mean = metric_data.mean()
                            
                            metrics_analysis.append({
                                'Métrica': metric,
                                'Média': f"{mean:.3f}",
                                'Q1 (25%)': f"{q25:.3f}",
                                'Mediana': f"{q50:.3f}",
                                'Q3 (75%)': f"{q75:.3f}"
                            })
                    
                    if metrics_analysis:
                        metrics_df = pd.DataFrame(metrics_analysis)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        st.markdown("""
                        <div class="insight-box">
                        <p><strong>Como interpretar:</strong></p>
                        <ul>
                        <li><strong>Q1 (25%):</strong> Bottom performers - necessita melhoria urgente</li>
                        <li><strong>Mediana:</strong> Performance padrão da posição</li>
                        <li><strong>Q3 (75%):</strong> Top performers - benchmark a atingir</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown(f"**Prioridades de Treinamento para {pos_key}:**")
                
                for i, priority in enumerate(pos_info['priorities'], 1):
                    st.markdown(f"{i}. {priority}")
                
                pos_performance = position_stats.loc[selected_pos, 'mean']
                overall_mean = filtered_data['Performance_Score'].mean()
                gap = pos_performance - overall_mean
                
                if gap > 0.05:
                    st.markdown(f"""
                    <div class="insight-box">
                    <h4>Performance Acima da Média</h4>
                    <p>{pos_key} está {gap:.3f} pontos <strong>acima</strong> da média geral ({overall_mean:.3f})</p>
                    <p><strong>Estratégia:</strong> Manter excelência focando em consistência nas métricas-chave</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif gap < -0.05:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>Oportunidade de Melhoria</h4>
                    <p>{pos_key} está {abs(gap):.3f} pontos <strong>abaixo</strong> da média geral ({overall_mean:.3f})</p>
                    <p><strong>Meta:</strong> Focar intensivamente nas métricas-chave listadas acima</p>
                    <p><strong>Benchmark:</strong> Atingir pelo menos o Q3 (75%) em cada métrica prioritária</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="insight-box">
                    <h4>Performance Equilibrada</h4>
                    <p>{pos_key} está próximo da média geral (diferença: {gap:+.3f})</p>
                    <p><strong>Estratégia:</strong> Identificar 1-2 métricas-chave para elevar ao nível Q3</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                other_positions_data = filtered_data[filtered_data['Position'] != selected_pos]['Performance_Score'].dropna()
                this_position_data = filtered_data[filtered_data['Position'] == selected_pos]['Performance_Score'].dropna()
                
                if len(this_position_data) > 1 and len(other_positions_data) > 1:
                    t_stat, p_value = stats.ttest_ind(this_position_data, other_positions_data)
                    
                    st.markdown("**Teste Estatístico: Diferença vs Outras Posições**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"Média {selected_pos}", f"{this_position_data.mean():.3f}")
                    with col2:
                        st.metric("Média Outras Posições", f"{other_positions_data.mean():.3f}")
                    
                    if p_value < 0.05:
                        st.success(f"Diferença estatisticamente significativa (p = {p_value:.4f})")
                    else:
                        st.info(f"Sem diferença estatística significativa (p = {p_value:.4f})")
            
            else:
                st.warning(f"Métricas específicas não disponíveis para {selected_pos}")
    
    st.markdown("---")
    st.markdown("### Análise Comparativa: Times Elite vs Times em Desenvolvimento")

    elite_teams = ['T1', 'Gen.G', 'JDG', 'BLG', 'G2 Esports'] 
    developing_teams = ['PaiN Gaming', 'MAD Lions KOI', 'PSG Talon']
    
    available_teams = filtered_data['TeamName'].unique()
    
    elite_available = [team for team in elite_teams if team in available_teams]
    developing_available = [team for team in developing_teams if team in available_teams]
    
    if len(elite_available) > 0 and len(developing_available) > 0:
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_elite = st.selectbox(
                "Time Elite (alta performance):",
                elite_available,
                index=0 if len(elite_available) > 0 else None
            )
        
        with col2:
            selected_developing = st.selectbox(
                "Time em Desenvolvimento:",
                developing_available,
                index=0 if len(developing_available) > 0 else None
            )
        
        if selected_elite and selected_developing:
            
            elite_data = filtered_data[filtered_data['TeamName'] == selected_elite]
            developing_data = filtered_data[filtered_data['TeamName'] == selected_developing]
            
            key_metrics = ['Win rate', 'KDA', 'DamagePercent', 'KP%', 'GoldPerMin', 'CSPerMin', 'Performance_Score']
            
            elite_stats = elite_data[key_metrics].mean()
            developing_stats = developing_data[key_metrics].mean()
            gap_analysis = elite_stats - developing_stats
            
            st.markdown(f"#### 📊 Comparação: {selected_elite} vs {selected_developing}")
            
            comparison_detailed = pd.DataFrame({
                f'{selected_elite} (Elite)': elite_stats.round(3),
                f'{selected_developing} (Desenvolvimento)': developing_stats.round(3),
                'Gap Absoluto': gap_analysis.round(3),
                'Gap Percentual (%)': ((gap_analysis / developing_stats) * 100).round(1),
                'Prioridade': ['🔴 CRÍTICA' if abs(gap) > developing_stats.loc[metric] * 0.2 
                              else '🟡 ALTA' if abs(gap) > developing_stats.loc[metric] * 0.1 
                              else '🟢 BAIXA' for metric, gap in gap_analysis.items()]
            })
            
            st.dataframe(comparison_detailed, use_container_width=True)
            
            critical_gaps = comparison_detailed[comparison_detailed['Prioridade'] == '🔴 CRÍTICA']
            high_gaps = comparison_detailed[comparison_detailed['Prioridade'] == '🟡 ALTA']
            
            if len(critical_gaps) > 0:
                st.markdown(f"""
                <div class="warning-box">
                <h4> GAPS CRÍTICOS IDENTIFICADOS</h4>
                <p><strong>Métricas com maior discrepância (>20%):</strong></p>
                <ul>
                {''.join([f'<li><strong>{metric}:</strong> {row["Gap Percentual (%)"]}% de diferença</li>' 
                         for metric, row in critical_gaps.iterrows()])}
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### Análise por Posição: Onde Melhorar Primeiro")
            
            position_comparison = []
            for position in ['Top', 'Jungle', 'Mid', 'Adc', 'Support']:
                elite_pos = elite_data[elite_data['Position'] == position]['Performance_Score'].mean()
                dev_pos = developing_data[developing_data['Position'] == position]['Performance_Score'].mean()
                
                if not pd.isna(elite_pos) and not pd.isna(dev_pos):
                    position_comparison.append({
                        'Posição': position,
                        'Elite Score': elite_pos,
                        'Desenvolvimento Score': dev_pos,
                        'Gap': elite_pos - dev_pos,
                        'Gap %': ((elite_pos - dev_pos) / dev_pos * 100) if dev_pos != 0 else 0
                    })
            
            if position_comparison:
                pos_df = pd.DataFrame(position_comparison).round(3)
                
                pos_df['Gap_Abs'] = pos_df['Gap'].abs()
                pos_df = pos_df.sort_values('Gap_Abs', ascending=False)
                
                st.dataframe(pos_df[['Posição', 'Elite Score', 'Desenvolvimento Score', 'Gap', 'Gap %']], use_container_width=True)

                worst_position = pos_df.iloc[0]['Posição'] if len(pos_df) > 0 else None
                worst_gap_pct = pos_df.iloc[0]['Gap %'] if len(pos_df) > 0 else 0
                
                best_position = pos_df.iloc[-1]['Posição'] if len(pos_df) > 0 else None
                best_gap_pct = pos_df.iloc[-1]['Gap %'] if len(pos_df) > 0 else 0
                
                if worst_position:
                    st.markdown(f"""
                    <div class="insight-box">
                    <h4>Plano de Desenvolvimento Prioritário</h4>
                    <p><strong>Posição com MAIOR Gap:</strong> {worst_position} ({abs(worst_gap_pct):.1f}% de diferença)</p>
                    <p><strong>Posição com MENOR Gap:</strong> {best_position} ({abs(best_gap_pct):.1f}% de diferença)</p>
                    
                    <p><strong>Interpretação:</strong></p>
                    <ul>
                    <li><strong>{worst_position}:</strong> {'Time elite está muito à frente' if pos_df.iloc[0]['Gap'] > 0 else 'Time desenvolvimento surpreendentemente melhor'}</li>
                    <li><strong>{best_position}:</strong> Posição mais equilibrada entre os times</li>
                    </ul>
                    
                    <p><strong>Estratégia Recomendada:</strong></p>
                    <ul>
                    <li>Focar recursos de coaching na posição {worst_position} (maior prioridade)</li>
                    <li>Estudar replays dos melhores jogadores {worst_position} dos times elite</li>
                    <li>Usar {best_position} como exemplo de processo de desenvolvimento eficaz</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            

            st.markdown("#### Soluções Específicas Baseadas nos Dados")
            
            solutions = []
            
            winrate_gap = gap_analysis.get('Win rate', 0)
            if winrate_gap > 0.1:
                solutions.append({
                    'Problema': 'Win Rate Baixo',
                    'Gap': f'{winrate_gap:.3f} ({(winrate_gap/developing_stats["Win rate"]*100):.1f}%)',
                    'Causa Raiz': 'Decisões táticas e teamplay',
                    'Solução Imediata': 'Melhorar comunicação e shot-calling',
                    'Solução Longo Prazo': 'Investir em analista dedicado para review de jogos'
                })
            
            kda_gap = gap_analysis.get('KDA', 0)
            if kda_gap > 0.5:
                solutions.append({
                    'Problema': 'KDA Baixo',
                    'Gap': f'{kda_gap:.3f} ({(kda_gap/developing_stats["KDA"]*100):.1f}%)',
                    'Causa Raiz': 'Positioning e sobrevivência em fights',
                    'Solução Imediata': 'Treinar positioning específico por role',
                    'Solução Longo Prazo': 'Bootcamp focado em teamfights'
                })
            
            gold_gap = gap_analysis.get('GoldPerMin', 0)
            cs_gap = gap_analysis.get('CSPerMin', 0)
            if gold_gap > 50 or cs_gap > 0.5:
                solutions.append({
                    'Problema': 'Eficiência Econômica',
                    'Gap': f'Gold: {gold_gap:.0f}/min, CS: {cs_gap:.2f}/min',
                    'Causa Raiz': 'Farm pattern e wave management',
                    'Solução Imediata': 'Drill de last-hitting e wave control',
                    'Solução Longo Prazo': 'Coach específico para macro game'
                })

            kp_gap = gap_analysis.get('KP%', 0)
            if kp_gap > 5:
                solutions.append({
                    'Problema': 'Baixa Participação em Kills',
                    'Gap': f'{kp_gap:.1f}%',
                    'Causa Raiz': 'Coordenação e map movement',
                    'Solução Imediata': 'Melhorar rotações e timings',
                    'Solução Longo Prazo': 'Sistema de comunicação padronizado'
                })
            
            if solutions:
                solutions_df = pd.DataFrame(solutions)
                st.dataframe(solutions_df, use_container_width=True)
                
                st.markdown("#### Timeline de Implementação Sugerido")
                
                st.markdown("""
                <div class="insight-box">
                <h4>Ações Imediatas (1-2 semanas)</h4>
                <ul>
                <li>Identificar os 2 gaps mais críticos da tabela acima</li>
                <li>Implementar drills específicos para essas áreas</li>
                <li>Estabelecer métricas de acompanhamento semanal</li>
                </ul>
                
                <h4>Médio Prazo (1-2 meses)</h4>
                <ul>
                <li>Contratar especialistas para as áreas identificadas</li>
                <li>Implementar sistema de análise de replays estruturado</li>
                <li>Estabelecer parcerias para scrimmages focadas</li>
                </ul>
                
                <h4>Longo Prazo (3-6 meses)</h4>
                <ul>
                <li>Reestruturação completa do staff técnico se necessário</li>
                <li>Investimento em infraestrutura de análise de dados</li>
                <li>Programa de intercâmbio com times de regiões mais fortes</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                

                total_gap = comparison_detailed['Gap Percentual (%)'].abs().mean()
                st.markdown(f"""
                <div class="warning-box">
                <h4>ROI Esperado</h4>
                <p><strong>Gap Médio Atual:</strong> {total_gap:.1f}%</p>
                <p><strong>Meta de Melhoria (6 meses):</strong> Reduzir gap para <5%</p>
                <p><strong>Indicadores de Sucesso:</strong></p>
                <ul>
                <li>Win Rate: +{winrate_gap*0.7:.2f} (70% do gap atual)</li>
                <li>Performance Score: +{gap_analysis.get('Performance_Score', 0)*0.6:.3f}</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("Selecione um time de cada categoria para realizar a análise comparativa.")
    
    else:
        available_teams_list = ", ".join(available_teams[:10])
        st.info(f"Times disponíveis no dataset atual: {available_teams_list}...")
        st.warning("Ajuste os filtros para incluir mais times ou verifique os nomes dos times no dataset.")


if __name__ == "__main__":
    main()
