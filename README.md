# 🎮 League of Legends Esports Analytics Dashboard

## 📋 Sobre o Projeto

Este projeto aplica técnicas avançadas de análise estatística e ciência de dados para explorar um dataset de jogadores profissionais de League of Legends do Mundial 2024. O objetivo é identificar padrões, tendências e propor soluções práticas baseadas em evidências estatísticas.

## 🎯 Objetivos

### Objetivo Geral
Aplicar técnicas de análise estatística para explorar dados de esports, identificar padrões e propor soluções práticas para melhoria de performance, apresentando resultados em um dashboard interativo.

### Objetivos Específicos
- ✅ **Análise Exploratória**: Identificação de padrões, outliers e tendências
- ✅ **Preparação de Dados**: Limpeza, tratamento de valores ausentes e engenharia de variáveis  
- ✅ **Modelagem Estatística**: Implementação de modelos de regressão linear
- ✅ **Validação Estatística**: Aplicação de testes de hipóteses e intervalos de confiança
- ✅ **Visualização Interativa**: Dashboard com gráficos e filtros dinâmicos
- ✅ **Soluções Práticas**: Recomendações baseadas em insights estatísticos

## 🗂️ Estrutura do Projeto

```
trabGiras/
├── 📊 player_statistics_cleaned_final.csv    # Dataset principal
├── 🎯 dashboard.py                            # Dashboard Streamlit
├── 📓 analise_completa.ipynb                  # Notebook Jupyter documentado
├── 📜 README.md                               # Documentação do projeto
└── 📋 script.py                               # Scripts auxiliares
```

## 📊 Dataset

**Fonte**: Estatísticas de jogadores profissionais do Mundial de League of Legends 2024

**Características**:
- 🎮 **83 jogadores** de 20+ times profissionais
- 🌍 **10+ países** representados  
- 🎭 **5 posições**: Top, Jungle, Mid, ADC, Support
- 📈 **25+ métricas** de performance individual e de equipe

### Principais Métricas
- **Performance Individual**: KDA, Win Rate, Kills/Deaths/Assists médios
- **Economia**: Gold per Minute, CS per Minute, Damage Percent
- **Equipe**: Kill Participation (KP%), First Blood %, Vision metrics
- **Early Game**: Gold/CS/XP difference at 15min

## 🛠️ Tecnologias Utilizadas

### Análise de Dados
- **Python 3.12+**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **SciPy** - Testes estatísticos

### Visualização
- **Streamlit** - Dashboard interativo
- **Plotly** - Gráficos interativos
- **Matplotlib/Seaborn** - Visualizações estáticas

### Modelagem Estatística
- **Scikit-learn** - Modelos de machine learning
- **Statsmodels** - Análise estatística avançada

## 🚀 Como Executar

### 1. Instalar Dependências
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn
```

### 2. Executar Dashboard
```bash
streamlit run dashboard.py
```

### 3. Acessar no Navegador
```
http://localhost:8501
```

## 📈 Funcionalidades do Dashboard

### 🏠 Visão Geral
- Estatísticas descritivas do dataset
- Distribuição de jogadores por posição e país
- Contexto e justificativa da análise

### 🔍 Análise Exploratória
- Identificação de outliers com boxplots interativos
- Top/Bottom performers por Performance Score
- Matriz de correlação com insights
- Análise comparativa por posição com gráfico radar

### 🧹 Preparação dos Dados
- Comparação antes/depois do tratamento
- Documentação da engenharia de variáveis
- Métricas de qualidade dos dados
- Distribuições das novas variáveis criadas

### 📊 Modelagem Estatística
- **Regressão Linear** configurável
- Seleção interativa de variáveis dependentes/independentes
- Métricas de performance (R², RMSE)
- Análise de resíduos e diagnóstico do modelo
- **Statsmodels** para análise estatística detalhada
- Predições interativas

### 🧪 Testes de Hipóteses
- **Teste t**: Diferença entre posições
- **Correlação**: Pearson e Spearman
- **ANOVA**: Diferenças entre múltiplas posições  
- **Normalidade**: Shapiro-Wilk e Kolmogorov-Smirnov
- Intervalos de confiança para todas as estimativas

### 📊 Visualizações Interativas
- **Scatter Plot Multi-dimensional**: 4 dimensões configuráveis
- **Heatmap de Times**: Performance normalizada
- **Gráfico Radar**: Comparação de jogadores
- **Violin Plots**: Distribuições por posição
- **Matriz de Scatter Plots**: Correlações visuais

### 💡 Soluções Práticas
- Recomendações estratégicas baseadas em top performers
- Guias específicos por posição
- Simulador de melhoria de performance
- Análise de ROI para times
- Discussão de limitações do estudo

## 📊 Principais Descobertas

### 🎯 Insights Estatísticos
- **KDA é o preditor mais forte** de Win Rate (r = 0.78, p < 0.001)
- **Kill Participation (KP%) > 70%** diferencia top performers
- **Diferenças significativas** entre posições confirmadas (ANOVA, p < 0.05)
- **Top 10 jogadores** mantêm KDA 45% superior à média

### 📈 Modelo Preditivo
- **R² = 0.653**: Explica 65.3% da variância em Win Rate
- **RMSE = 0.089**: Erro médio de 8.9%
- **Principais variáveis**: KDA, Damage%, KP%, Gold/min
- **Diagnósticos**: Homocedasticidade confirmada

### 💡 Recomendações Práticas

#### Por Posição:
- **Top**: Foco em farm early game e teleports eficientes
- **Jungle**: Maximizar map presence e coordenação de objetivos
- **Mid**: Balancear farm com roaming e teamfight damage  
- **ADC**: Positioning para DPS e eficiência econômica
- **Support**: Vision control e roaming inteligente

#### Gerais:
1. **Priorizar KDA** como métrica principal de avaliação
2. **Maximizar Kill Participation** em teamfights
3. **Otimizar eficiência econômica** (Gold/CS per minute)
4. **Desenvolver programas de treinamento específicos** por role

## 🔬 Metodologia Estatística

### Análise Exploratória
- Estatísticas descritivas completas
- Identificação de outliers (método IQR)
- Análise de correlações (Pearson)
- Visualizações multivariadas

### Modelagem
- **Regressão Linear Múltipla**
- Divisão train/test (70/30)
- Validação cruzada
- Análise de resíduos

### Testes de Hipóteses
- **Teste t independente** (α = 0.05)
- **ANOVA One-Way** para múltiplos grupos
- **Testes de correlação** (Pearson e Spearman)
- **Intervalos de confiança 95%**

### Diagnósticos
- Teste de normalidade (Shapiro-Wilk)
- Homocedasticidade (Breusch-Pagan)
- Análise de multicolinearidade (VIF)
- Q-Q plots para validação

## ⚠️ Limitações

- **Temporal**: Análise baseada em snapshot (Mundial 2024)
- **Meta Game**: Mudanças de patches podem afetar relevância
- **Amostra**: Alguns times/posições com poucos jogadores
- **Causalidade**: Correlações não implicam relações causais
- **Fatores Externos**: Coaching e ambiente de equipe não considerados

## 👥 Autores

**Rafael Ferreira** - Análise de Dados e Estatística  
📧 Email: [rafabf18@gmail.com]  
🔗 GitHub: [@RafaelFerreira18](https://github.com/RafaelFerreira18)

**** - Análise de Dados e Estatística  
📧 Email: [pauloricardoctec@gmail.com]  
🔗 GitHub: [@PauloRCunhaDev](https://github.com/PauloRCunhaDev)
