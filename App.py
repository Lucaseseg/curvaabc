import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
import google.generativeai as genai
import time
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import re
from streamlit_modal import Modal
import markdown  # Import para conversão de Markdown

# Configuração inicial da página
st.set_page_config(page_title="Análise Curva ABC", page_icon="📊", layout="wide")

# Configuração da API do Google Gemini
api_key = st.secrets["GOOGLE_API_KEY"]  # Ou st.secrets["google"]["api_key"] se usar estrutura aninhada

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Chave API do Google Gemini não encontrada. Por favor, configure a variável GOOGLE_API_KEY nas Secrets do Streamlit.")

# CSS Personalizado para estilização
def custom_css():
    st.markdown("""
    <style>
    /* Barra Lateral */
    [data-testid="stSidebar"] {
        background-color: #f0f0f0; /* Cor neutra */
        color: #333333; /* Texto escuro para contraste */
        padding: 20px;
    }

    /* Fundo da Página */
    .css-1d391kg { 
        background-color: #F4F6F7; /* Fundo claro e neutro */
        margin: 0;
        padding: 0;
    }

    /* Rodapé */
    .footer {
        position: fixed;
        right: 20px;
        bottom: 20px;
        font-size: 24px;
        color: #27AE60;
    }

    .footer .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    .footer .tooltip .tooltiptext {
        visibility: hidden;
        width: 160px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .footer .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #555 transparent transparent transparent;
    }

    .footer .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Botões Personalizados */
    .stButton>button {
        background-color: #27AE60; /* Verde mais neutro */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: #1E8449; /* Verde escuro */
    }

    /* Cabeçalhos */
    .header-title {
        color: #27AE60;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Tabelas */
    .dataframe {
        border: 2px solid #27AE60;
        border-radius: 10px;
    }

    /* Pop-Up */
    .modal-content {
        background-color: #fff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }

    .modal-header {
        text-align: center;
        margin-bottom: 20px;
    }

    .modal-header h2 {
        color: #27AE60;
    }

    .modal-body p {
        color: #34495E;
        font-size: 16px;
        line-height: 1.6;
    }

    .modal-footer {
        text-align: center;
        margin-top: 20px;
    }

    .modal-footer button {
        background-color: #27AE60;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }

    .modal-footer button:hover {
        background-color: #1E8449;
    }
    </style>
    """, unsafe_allow_html=True)

custom_css()

# Função para exibir o logo na barra lateral
def exibir_logo():
    img_path = "eseg.png"  # Substitua pelo caminho da sua imagem
    if os.path.exists(img_path):
        st.sidebar.image(img_path, caption="ESEG Corp", use_column_width=True)
    else:
        st.sidebar.warning("Logo não encontrado!")

# Função para carregar a planilha
def carregar_planilha(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar a planilha: {e}")
    st.sidebar.error("Formato não suportado. Use .csv ou .xlsx.")
    return None

# Função para converter markdown para ReportLab
def markdown_para_reportlab(texto):
    # Converte Markdown para HTML
    html = markdown.markdown(texto)

    # Substituir listas não ordenadas e ordenadas por tags compatíveis
    html = html.replace('<ul>', '<bullet>').replace('</ul>', '</bullet>')
    html = html.replace('<ol>', '<ordered>').replace('</ol>', '</ordered>')
    html = html.replace('<li>', '<bullet>&bull; ').replace('</li>', '</bullet>')

    # Remover outras tags HTML
    clean_text = re.sub('<[^<]+?>', '', html)

    return clean_text

# Função para calcular Curva ABC
def calcular_curva_abc(df):
    df = df.copy()
    df['Valor Total'] = df['Preço'] * df['Quantidade']
    df_sorted = df.sort_values(by='Valor Total', ascending=False).reset_index(drop=True)
    df_sorted['% Acumulado'] = df_sorted['Valor Total'].cumsum() / df_sorted['Valor Total'].sum() * 100
    df_sorted['Classe'] = 'C'
    df_sorted.loc[df_sorted['% Acumulado'] <= 80, 'Classe'] = 'A'
    df_sorted.loc[(df_sorted['% Acumulado'] > 80) & (df_sorted['% Acumulado'] <= 95), 'Classe'] = 'B'
    return df_sorted

# Função para preprocessar os dados
def preprocessar_dados(df):
    df = df.dropna(subset=["Preço", "Quantidade"])
    z_scores = np.abs(stats.zscore(df[["Preço", "Quantidade"]]))
    df = df[(z_scores < 3).all(axis=1)]
    return df

# Função para determinar o número ótimo de clusters
def determinar_n_clusters(df, max_clusters=10):
    scaler = StandardScaler()
    features = ["Preço", "Quantidade"]
    scaled_features = scaler.fit_transform(df[features])

    sse = []
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(score)

    # Sugestão de Número de Clusters
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    st.success(f"Número ótimo de clusters sugerido: *{best_k}*")

    # Plot Método do Cotovelo e Silhouette Score
    fig, ax1 = plt.subplots(figsize=(8, 4))  # Reduzido o tamanho para melhor ajuste
    color = '#27AE60'
    ax1.set_xlabel('Número de Clusters')
    ax1.set_ylabel('SSE', color=color)
    ax1.plot(range(2, max_clusters + 1), sse, marker='o', color=color, label='SSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = '#E67E22'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='x', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Método do Cotovelo e Silhouette Score', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    return best_k

# Função para aplicar K-Means e retornar os centróides e DataFrame clusterizado
def aplicar_kmeans(df, n_clusters=3):
    scaler = StandardScaler()
    features = ["Preço", "Quantidade"]
    scaled_features = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_features)

    # Centróides na escala original
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=features)
    centroids_df['Cluster'] = range(n_clusters)

    return df, centroids_df

# Função para visualizar distribuição das classes ABC nos clusters e retornar a crosstab
def visualizar_abc_clusters(df):
    crosstab = pd.crosstab(df['Classe'], df['Cluster'])
    return crosstab

# Função para gerar análise da curva ABC
def gerar_analise_gemini(df):
    produtos_str = df[['Nome', 'Quantidade', 'Preço', 'Classe', 'Cluster']].to_string(index=False)
    data_atual = time.strftime('%d/%m/%Y às %H:%M')  # Captura a data e hora atual

    # Prompt fixo para análise
    prompt = (
        f"Data e Hora do Relatório: {data_atual}\n\n"
        f"A partir de agora você é um especialista em Supply Chain. "
        f"Faça uma análise detalhada desses produtos com base na curva ABC e nos clusters identificados. "
        f"Explique o que os clusters indicam sobre o comportamento dos produtos e forneça insights "
        f"sobre estoque, faturamento e possíveis sinergias entre produtos do mesmo cluster. "
        f"Seja breve, porém seja extremamente específico com os produtos a seguir:\n{produtos_str}"
    )

    model = genai.GenerativeModel("gemini-1.5-flash")

    # Exibir o spinner antes de gerar a resposta
    with st.spinner("Gerando análise..."):
        response = model.generate_content(prompt)

        # Simulação de digitação da resposta
        resposta_texto = response.text
        resposta_container = st.empty()  # Cria um espaço para a resposta

        # Simulação do efeito de digitação
        for i in range(len(resposta_texto) + 1):
            resposta_container.markdown(resposta_texto[:i], unsafe_allow_html=True)
            time.sleep(0.005)  # Ajuste o tempo para controlar a velocidade da digitação

    # Salvar a análise gerada no estado da sessão
    st.session_state.analise_gemini = resposta_texto

# Função para adicionar produtos manualmente com callback
def adicionar_produto():
    nome = st.session_state.nome_produto
    preco = st.session_state.preco_produto
    quantidade = st.session_state.quantidade_produto

    if nome and preco > 0 and quantidade > 0:
        if 'produtos' not in st.session_state:
            st.session_state.produtos = []
        st.session_state.produtos.append(
            {"Nome": nome, "Preço": preco, "Quantidade": quantidade}
        )
        st.sidebar.success(f"Produto *{nome}* adicionado com sucesso!")
        # Limpar os campos após adicionar
        st.session_state.nome_produto = ""
        st.session_state.preco_produto = 0.0
        st.session_state.quantidade_produto = 0
    else:
        st.sidebar.error("Preencha todos os campos corretamente.")

# Função para adicionar produtos manualmente
def adicionar_produto_manual():
    with st.sidebar.expander("📋 Gerenciamento de Produtos", expanded=False):
        st.markdown("### Adicionar Produto Manualmente")
        nome = st.text_input("📝 Nome do Produto", key="nome_produto")
        preco = st.number_input("💰 Preço (R$)", min_value=0.0, format="%.2f", key="preco_produto")
        quantidade = st.number_input("📦 Quantidade", min_value=0, step=1, key="quantidade_produto")

        st.button("➕ Adicionar Produto", key="botao_adicionar", on_click=adicionar_produto)

# Funções para salvar gráficos em imagens
def salvar_grafico_pizza(data, titulo):
    buffer = BytesIO()
    fig, ax = plt.subplots(figsize=(3, 3))  # Tamanho reduzido para melhor ajuste
    ax.pie(
        data,
        labels=data.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("pastel")[:len(data)],
        wedgeprops={'edgecolor': 'white'}
    )
    ax.set_title(titulo, fontsize=12, color="#27AE60")
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(buffer, format="PNG")
    buffer.seek(0)
    plt.close(fig)
    return buffer

def salvar_grafico_dispersao(df):
    buffer = BytesIO()
    fig, ax = plt.subplots(figsize=(4, 3))  # Tamanho reduzido para melhor ajuste
    sns.scatterplot(
        x='Quantidade', y='Preço', data=df, hue='Cluster', palette='deep', s=60, edgecolor='white', alpha=0.7, ax=ax
    )
    plt.title('Dispersão de Preço x Quantidade com Clusters', fontsize=12, color="#27AE60")
    plt.xlabel('Quantidade', fontsize=10)
    plt.ylabel('Preço (R$)', fontsize=10)
    plt.legend(title='Cluster', fontsize=8, title_fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(buffer, format="PNG")
    buffer.seek(0)
    plt.close(fig)
    return buffer

# Função para gerar PDF com gráficos e análise formatada
def gerar_pdf(df, class_counts, class_counts2, analise_texto):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=18
    )
    elements = []

    # Define os estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CenterTitle',
        parent=styles['Title'],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=20,
        spaceBefore=20
    ))
    styles.add(ParagraphStyle(
        name='SectionHeader',
        fontSize=18,
        spaceAfter=10,
        textColor=colors.HexColor("#27AE60"),
        leading=22,
        alignment=TA_LEFT
    ))
    # Renomear 'Heading1' e 'Heading2' para evitar conflito
    styles.add(ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceAfter=12,
        textColor=colors.HexColor("#27AE60")
    ))
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=16,
        leading=20,
        spaceAfter=10,
        textColor=colors.HexColor("#27AE60")
    ))
    styles.add(ParagraphStyle(
        name='NormalLeft',
        alignment=TA_LEFT,
        fontSize=12,
        spaceAfter=10,
        leading=15
    ))
    # Renomear o estilo para 'CustomBullet' para evitar conflito
    styles.add(ParagraphStyle(
        name='CustomBullet',
        parent=styles['Normal'],
        leftIndent=20,
        bulletIndent=10,
        bulletFontName='Helvetica',
        bulletFontSize=12,
    ))

    # Capa
    if os.path.exists("eseg.png"):
        logo = "eseg.png"
        im = Image(logo, 3 * inch, 0.8 * inch)
        im.hAlign = 'CENTER'
        elements.append(im)
        elements.append(Spacer(1, 12))

    # Título na capa
    elements.append(Paragraph("Relatório de Análise da Curva ABC", styles['CenterTitle']))
    elements.append(Spacer(1, 50))
    elements.append(PageBreak())

    # Sumário Manual
    elements.append(Paragraph("Sumário", styles['SectionHeader']))
    elements.append(Spacer(1, 12))
    sumario = """
    1. Introdução ......................................... 1
    2. Metodologia ........................................ 2
    3. Estatísticas Descritivas ........................... 3
    4. Tabela de Produtos ................................. 4
    5. Gráficos ........................................... 5
    6. Detalhamento dos Clusters .......................... 6
    7. Análise Detalhada com Google Gemini ................ 7
    8. Recomendações ...................................... 8
    9. Referências ........................................ 9
    """
    for linha in sumario.strip().split('\n'):
        elements.append(Paragraph(linha.strip(), styles['NormalLeft']))
    elements.append(PageBreak())

    # Sumário Executivo
    elements.append(Paragraph("Sumário Executivo", styles['SectionHeader']))
    elements.append(Paragraph(
        "Este relatório apresenta uma análise detalhada da Curva ABC dos produtos, incluindo a clusterização utilizando o método K-Means. Os principais objetivos são identificar produtos de alta importância, otimizar o gerenciamento de estoque e fornecer insights estratégicos para melhorar o desempenho da empresa.",
        styles['NormalLeft']
    ))
    elements.append(PageBreak())

    # **Introdução**
    elements.append(Paragraph("Introdução", styles['SectionHeader']))
    elements.append(Paragraph(
        "A Curva ABC é uma técnica amplamente utilizada na gestão de estoques e processos, permitindo a classificação dos itens com base em sua importância relativa. Através dessa análise, é possível identificar quais produtos são mais relevantes para o faturamento e foco estratégico da empresa. Este relatório tem como objetivo aplicar a análise da Curva ABC aos produtos listados, fornecendo insights que auxiliem na tomada de decisões gerenciais.",
        styles['NormalLeft']
    ))
    elements.append(PageBreak())

    # **Metodologia**
    elements.append(Paragraph("Metodologia", styles['SectionHeader']))
    elements.append(Paragraph(
        "Para a elaboração deste relatório, adotamos a seguinte metodologia:",
        styles['NormalLeft']
    ))
    metodologia = [
        "1. **Coleta de Dados:** Reunimos informações sobre os produtos, incluindo nome, preço unitário e quantidade em estoque.",
        "2. **Cálculo do Valor Total:** Calculamos o valor total de cada produto multiplicando o preço unitário pela quantidade.",
        "3. **Ordenação Decrescente:** Organizamos os produtos em ordem decrescente de valor total para identificar os itens de maior impacto financeiro.",
        "4. **Cálculo do Percentual Acumulado:** Determinamos o percentual acumulado de cada produto em relação ao valor total acumulado de todos os produtos.",
        "5. **Classificação ABC:** Classificamos os produtos em classes A, B ou C com base nos seguintes critérios:",
        "   - **Classe A:** Itens que representam até 80% do valor acumulado.",
        "   - **Classe B:** Itens que representam entre 80% e 95% do valor acumulado.",
        "   - **Classe C:** Itens que representam os 5% restantes do valor acumulado.",
        "6. **Análise de Clusterização (K-Means):** Aplicamos o algoritmo K-Means para identificar agrupamentos naturais entre os produtos, considerando as variáveis de preço e quantidade.",
        "7. **Interpretação dos Resultados:** Analisamos os clusters e as classes ABC para extrair insights sobre o comportamento dos produtos e oportunidades de otimização.",
    ]
    # Texto detalhado para cada tópico
    metodologia_detalhada = {
        "1": """
Reunimos informações detalhadas sobre os produtos comercializados pela empresa. Essa etapa é crucial para garantir que a análise seja baseada em dados precisos e abrangentes. Os dados coletados incluem:

- **Nome do Produto:** Identificação única de cada item, permitindo a distinção clara entre os diferentes produtos no portfólio.
- **Preço Unitário:** Valor monetário pelo qual cada unidade do produto é vendida. Este dado é essencial para o cálculo do faturamento potencial e da margem de lucro.
- **Quantidade em Estoque:** Número de unidades disponíveis de cada produto. Essa informação é fundamental para avaliar a disponibilidade do produto para vendas futuras e para o planejamento de reposição de estoque.
        """,
        "2": """
Nesta etapa, calculamos o valor total que cada produto representa no estoque, multiplicando o preço unitário pela quantidade em estoque.

Este cálculo permite identificar o peso financeiro de cada produto no inventário total da empresa. Produtos com alto valor total podem indicar itens de alta rotatividade ou produtos de alto custo que exigem atenção especial na gestão.
        """,
        "3": """
Após calcular o valor total de cada produto, organizamos os itens em ordem decrescente com base nesse valor. Esta ordenação facilita a visualização dos produtos que mais contribuem para o valor total do estoque. Ao ordenar os produtos desta forma, conseguimos:

- Identificar rapidamente os produtos de maior impacto financeiro.
- Priorizar a análise e gestão dos itens mais relevantes.
- Estabelecer uma base para a classificação ABC.

A ordenação é feita utilizando ferramentas de análise de dados que permitem a manipulação eficiente de grandes volumes de informações.
        """,
        "4": """
Com os produtos ordenados, calculamos o percentual acumulado do valor total para cada produto em relação ao valor total acumulado de todos os produtos.

Este percentual acumulado ajuda a compreender como cada produto contribui para o valor total do estoque e é fundamental para a definição das classes A, B e C na etapa seguinte.
        """,
        "5": """
Com base nos percentuais acumulados, classificamos os produtos em três categorias principais, seguindo a metodologia da Curva ABC:

- **Classe A:** Produtos que representam aproximadamente os primeiros 80% do valor acumulado. Geralmente, constituem cerca de 20% dos itens em quantidade, mas são os mais valiosos em termos financeiros. Esses produtos requerem um gerenciamento rigoroso de estoque, previsões de demanda precisas e atenção especial em termos de qualidade e disponibilidade.
- **Classe B:** Produtos que contribuem com os próximos 15% do valor acumulado, totalizando até 95% quando somados aos da Classe A. Correspondem a uma parcela maior em quantidade, mas com menor impacto individual no valor total. Devem ser monitorados regularmente, com foco em otimização de estoque e melhoria de eficiência.
- **Classe C:** Produtos que representam os últimos 5% do valor acumulado. Apesar de serem numerosos (podendo chegar a 50% dos itens em quantidade), têm baixo impacto financeiro. A gestão desses itens pode ser simplificada para reduzir custos operacionais, evitando excesso de estoque e obsolescência.

A classificação é aplicada conforme os seguintes critérios:

- **Classe A:** Percentual acumulado de 0% a 80%.
- **Classe B:** Percentual acumulado acima de 80% até 95%.
- **Classe C:** Percentual acumulado acima de 95% até 100%.

Essa categorização permite alocar recursos de forma eficiente, focando nos produtos que mais influenciam o desempenho financeiro da empresa.
        """,
        "6": """
Para aprofundar a análise e identificar padrões ocultos nos dados, aplicamos o algoritmo de clusterização K-Means. Este método agrupa os produtos com características similares, considerando múltiplas variáveis. O processo envolve:

- **Seleção das Variáveis de Análise:** Utilizamos as variáveis 'Preço Unitário' e 'Quantidade em Estoque' para capturar tanto o valor monetário quanto a disponibilidade de cada produto.
- **Normalização dos Dados:** Padronizamos as variáveis para eliminar diferenças de escala que possam influenciar os resultados. A normalização é feita transformando os dados para que tenham média zero e desvio padrão um.
- **Determinação do Número Ótimo de Clusters:** Utilizamos métodos estatísticos, como o método do cotovelo (Elbow Method) e o coeficiente de silhueta (Silhouette Score), para definir o número adequado de clusters que melhor segmenta os dados sem super ou subagrupamentos.
- **Aplicação do Algoritmo K-Means:** Com o número de clusters definido, aplicamos o K-Means para segmentar os produtos. O algoritmo atribui cada produto ao cluster cujo centroide (ponto médio) é o mais próximo, minimizando a variabilidade dentro dos clusters.
- **Análise dos Clusters Formados:** Avaliamos as características de cada cluster, como médias e dispersões, para interpretar os grupos formados. Isso nos ajuda a identificar segmentos de produtos com comportamentos semelhantes.

A clusterização complementa a análise ABC, oferecendo uma visão multidimensional dos produtos e auxiliando na elaboração de estratégias específicas para cada grupo.
        """,
        "7": """
Com os produtos classificados e agrupados, procedemos à interpretação detalhada dos resultados:

- **Análise da Distribuição das Classes ABC nos Clusters:** Verificamos como os produtos das classes A, B e C estão distribuídos entre os diferentes clusters. Isso pode revelar, por exemplo, se produtos de alto valor (Classe A) estão concentrados em determinados clusters.
- **Identificação de Padrões e Tendências:** Avaliamos se existem tendências, como produtos de baixo preço e alta quantidade em um cluster específico, ou produtos de alto preço e baixa quantidade em outro. Essas informações podem indicar segmentos de mercado ou comportamentos de consumo.
- **Insights para Gestão de Estoque:** Compreendemos quais clusters requerem maior atenção em termos de reposição de estoque, negociação com fornecedores ou estratégias de precificação.
- **Oportunidades de Sinergia e Otimização:** Identificamos possibilidades de agrupar produtos para promoções conjuntas, otimizar logística de armazenamento ou ajustar o mix de produtos oferecidos.
- **Avaliação de Riscos:** Reconhecemos produtos ou clusters que possam representar riscos, como excesso de estoque em itens de baixa rotatividade ou dependência excessiva de poucos produtos para o faturamento total.

A interpretação dos resultados é fundamental para transformar a análise em ações estratégicas. Envolve a colaboração entre diferentes áreas da empresa, como logística, vendas, marketing e finanças, para alinhar as decisões aos objetivos organizacionais.
        """
    }

    for passo in metodologia:
        match = re.match(r'^(\d+)\.\s(.*)', passo.strip())
        if match:
            num = match.group(1)
            title = match.group(2)
            elements.append(Paragraph(f"<b>{num}. {title}</b>", styles['NormalLeft']))
            elements.append(Spacer(1, 6))
            if num in metodologia_detalhada:
                detalhe = metodologia_detalhada[num]
                # Processar formatações dentro do texto
                detalhe = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', detalhe)
                detalhe = re.sub(r'\*(.*?)\*', r'<i>\1</i>', detalhe)
                # Adicionar quebras de linha
                detalhe = detalhe.replace('\n', '<br/>')
                elements.append(Paragraph(detalhe, styles['NormalLeft']))
                elements.append(Spacer(1, 12))
        else:
            # Processar subitens (por exemplo, itens com '   - ')
            subitem_match = re.match(r'^\s+-\s(.*)', passo.strip())
            if subitem_match:
                subtext = subitem_match.group(1)
                subtext = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', subtext)
                subtext = re.sub(r'\*(.*?)\*', r'<i>\1</i>', subtext)
                elements.append(Paragraph(subtext, styles['NormalLeft'], bulletText='•'))
                elements.append(Spacer(1, 6))
            else:
                elements.append(Paragraph(passo.strip(), styles['NormalLeft']))
                elements.append(Spacer(1, 6))
    elements.append(PageBreak())

    # Estatísticas Descritivas
    elements.append(Paragraph("Estatísticas Descritivas", styles['SectionHeader']))
    descr = df[['Preço', 'Quantidade']].describe().round(2)
    data = [['Métrica', 'Preço (R$)', 'Quantidade']]
    for index, row in descr.iterrows():
        data.append([index, row['Preço'], row['Quantidade']])
    table = Table(data, hAlign='LEFT')
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#27AE60")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ])
    table.setStyle(table_style)

    # Alternar cores das linhas
    for i in range(1, len(data)):
        bg_color = colors.lightgrey if i % 2 == 0 else colors.whitesmoke
        table_style.add('BACKGROUND', (0, i), (-1, i), bg_color)
    table.setStyle(table_style)

    elements.append(table)
    elements.append(PageBreak())

    # Tabela de Produtos Detalhada
    elements.append(Paragraph("Tabela de Produtos", styles['SectionHeader']))
    data = [['Nome', 'Preço (R$)', 'Quantidade', 'Valor Total (R$)', '% Acumulado', 'Classe', 'Cluster']]
    for index, row in df.iterrows():
        data.append([
            row['Nome'],
            f"{row['Preço']:.2f}",
            row['Quantidade'],
            f"{row['Valor Total']:.2f}",
            f"{row['% Acumulado']:.2f}%",
            row['Classe'],
            row['Cluster']
        ])

    table = Table(data, hAlign='LEFT', repeatRows=1)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#27AE60")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ])
    table.setStyle(table_style)

    # Alternar cores das linhas
    for i in range(1, len(data)):
        bg_color = colors.lightgrey if i % 2 == 0 else colors.whitesmoke
        table_style.add('BACKGROUND', (0, i), (-1, i), bg_color)
    table.setStyle(table_style)

    elements.append(table)
    elements.append(PageBreak())

    # Gráficos
    elements.append(Paragraph("Gráficos", styles['SectionHeader']))

    # Distribuição das Classes
    elements.append(Paragraph("Distribuição das Classes (A, B, C)", styles['SectionHeader']))
    grafico_pizza_classe = salvar_grafico_pizza(class_counts, "Distribuição das Classes (A, B, C)")
    im_pizza_classe = Image(grafico_pizza_classe, 3 * inch, 3 * inch)  # Tamanho reduzido
    im_pizza_classe.hAlign = 'CENTER'
    elements.append(im_pizza_classe)
    elements.append(Spacer(1, 12))

    # Distribuição dos Clusters
    elements.append(Paragraph("Distribuição dos Clusters", styles['SectionHeader']))
    grafico_pizza_cluster = salvar_grafico_pizza(class_counts2, "Distribuição dos Clusters")
    im_pizza_cluster = Image(grafico_pizza_cluster, 3 * inch, 3 * inch)  # Tamanho reduzido
    im_pizza_cluster.hAlign = 'CENTER'
    elements.append(im_pizza_cluster)
    elements.append(PageBreak())

    # Gráfico de Dispersão
    elements.append(Paragraph("Gráfico de Dispersão de Clusters", styles['SectionHeader']))
    grafico_dispersao = salvar_grafico_dispersao(df)
    im_dispersao = Image(grafico_dispersao, 4 * inch, 3 * inch)  # Tamanho reduzido
    im_dispersao.hAlign = 'CENTER'
    elements.append(im_dispersao)
    elements.append(PageBreak())

    # Detalhamento dos Clusters
    elements.append(Paragraph("Detalhamento dos Clusters", styles['SectionHeader']))
    centroids = df.groupby('Cluster').agg({
        'Preço': 'mean',
        'Quantidade': 'mean',
        'Valor Total': 'mean'
    }).round(2)
    centroids.reset_index(inplace=True)
    data = [['Cluster', 'Preço Médio (R$)', 'Quantidade Média', 'Valor Total Médio (R$)']]
    for index, row in centroids.iterrows():
        data.append([
            row['Cluster'],
            f"{row['Preço']:.2f}",
            row['Quantidade'],
            f"{row['Valor Total']:.2f}"
        ])
    table = Table(data, hAlign='LEFT')
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#27AE60")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ])
    table.setStyle(table_style)

    # Alternar cores das linhas
    for i in range(1, len(data)):
        bg_color = colors.lightgrey if i % 2 == 0 else colors.whitesmoke
        table_style.add('BACKGROUND', (0, i), (-1, i), bg_color)
    table.setStyle(table_style)

    elements.append(table)
    elements.append(PageBreak())

    # Análise Detalhada
    elements.append(Paragraph("Análise Detalhada com Google Gemini", styles['SectionHeader']))

    # Processar 'analise_texto' para aplicar formatação
    lines = analise_texto.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 12))
            continue

        # Processar cabeçalhos
        if line.startswith('## '):
            header_text = line[3:].strip()
            elements.append(Paragraph(header_text, styles['CustomHeading2']))
        elif line.startswith('# '):
            header_text = line[2:].strip()
            elements.append(Paragraph(header_text, styles['CustomHeading1']))
        else:
            # Processar negrito '**texto**' e itálico '*texto*'
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
            # Processar listas não ordenadas
            if line.startswith('- '):
                bullet_text = line[2:].strip()
                elements.append(Paragraph(bullet_text, styles['NormalLeft'], bulletText='•'))
                elements.append(Spacer(1, 6))
            # Processar listas ordenadas
            elif re.match(r'^\d+\.\s', line):
                match = re.match(r'^(\d+)\.\s(.*)', line)
                num = match.group(1)
                text = match.group(2)
                elements.append(Paragraph(text.strip(), styles['NormalLeft'], bulletText=f'{num}.'))
                elements.append(Spacer(1, 6))
            else:
                # Parágrafo normal
                elements.append(Paragraph(line, styles['NormalLeft']))
                elements.append(Spacer(1, 6))
    elements.append(PageBreak())

    # Recomendações
    elements.append(Paragraph("Recomendações", styles['SectionHeader']))
    elements.append(Paragraph(
        "Com base na análise realizada, recomendamos as seguintes ações para otimizar o gerenciamento de estoque e aumentar o faturamento:",
        styles['NormalLeft']
    ))
    elements.append(Spacer(1, 12))
    recomendacoes = [
        "1. *Foco nos Produtos Classe A:* Priorizar o gerenciamento e controle de estoque dos produtos classificados como Classe A, pois representam a maior parte do valor total.",
        "2. *Promoções para Produtos Classe B:* Implementar estratégias de marketing e promoções para os produtos Classe B a fim de aumentar sua contribuição para o valor total.",
        "3. *Redução de Estoque de Classe C:* Considerar a redução do estoque ou descontinuação dos produtos Classe C que não contribuem significativamente para o faturamento.",
        "4. *Sinergias entre Clusters:* Identificar produtos dentro dos mesmos clusters que possam ser vendidos em conjunto para aumentar as vendas cruzadas.",
        "5. *Revisão Periódica:* Realizar análises periódicas da Curva ABC e dos clusters para ajustar as estratégias conforme as mudanças no comportamento de compra dos clientes."
    ]

    for rec in recomendacoes:
        match = re.match(r'^(\d+)\.\s(.*)', rec.strip())
        if match:
            num = match.group(1)
            text = match.group(2)
            # Processar itálico dentro do texto
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            elements.append(Paragraph(text, styles['NormalLeft'], bulletText=f'{num}.'))
            elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph(rec.strip(), styles['NormalLeft']))
            elements.append(Spacer(1, 6))
    elements.append(PageBreak())

    # Referências
    elements.append(Paragraph("Referências", styles['SectionHeader']))
    referencias = [
        "- Metodologia ABC: https://pt.wikipedia.org/wiki/An%C3%A1lise_ABC",
        "- K-Means Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
        "- Ballou, R. H. (2006). Gerenciamento da Cadeia de Suprimentos/Logística Empresarial. Bookman.",
        "- Slack, N., Chambers, S., & Johnston, R. (2009). Administração da Produção. Atlas.",
        "- Chopra, S., & Meindl, P. (2016). Gerenciamento da Cadeia de Suprimentos: Estratégia, Planejamento e Operação. Pearson.",
        "- Análise ABC: https://pt.wikipedia.org/wiki/An%C3%A1lise_ABC",
        "- Supply Chain Management: https://pt.wikipedia.org/wiki/Gest%C3%A3o_da_cadeia_de_suprimentos"
    ]

    for ref in referencias:
        elements.append(Paragraph(ref.strip('- '), styles['NormalLeft'], bulletText='•'))
        elements.append(Spacer(1, 6))
    elements.append(PageBreak())

    # Rodapé com data e número de página
    def add_footer(canvas_obj, doc_obj):
        page_num = canvas_obj.getPageNumber()
        page_text = f"Página {page_num}"
        date_text = f"Data: {time.strftime('%d/%m/%Y às %H:%M')}"

        canvas_obj.setFont('Helvetica', 10)
        canvas_obj.setFillColor(colors.grey)

        # Posiciona a data no centro, na parte inferior da página
        canvas_obj.drawCentredString(A4[0] / 2.0, 0.5 * inch, date_text)
        # Posiciona o número da página abaixo da data
        canvas_obj.drawCentredString(A4[0] / 2.0, 0.35 * inch, page_text)

    # Construção do documento com o novo rodapé
    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer.getvalue()

# Função para baixar o PDF no Streamlit (sidebar)
def baixar_pdf_sidebar():
    if 'pdf' in st.session_state:
        st.sidebar.download_button(
            label="📥 Baixar Relatório em PDF",
            data=st.session_state['pdf'],
            file_name="relatorio_curva_abc.pdf",
            mime="application/pdf",
            key='download_pdf_sidebar'
        )
    else:
        st.sidebar.info("Gerar o relatório na aba 'Análise Gemini' para disponibilizar o download.")

# Função para adicionar o rodapé com ícone e tooltip usando Emoji
def adicionar_footer():
    footer_html = """
    <div class="footer">
        <div class="tooltip">
            ⚡  
            <span class="tooltiptext">Powered by ALJ Corp</span>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Função para exibir o pop-up de boas-vindas usando streamlit-modal
def exibir_pop_up():
    if 'show_modal' not in st.session_state:
        st.session_state.show_modal = True

    modal = Modal(title="Bem-vindo ao Análise Curva ABC! 🎉", key="welcome_modal")
    if st.session_state.show_modal:
        with modal.container():
            # Corpo do Modal
            st.markdown("""
                <div class='modal-body'>
                    <p>
                        Este aplicativo foi desenvolvido para auxiliar na análise da Curva ABC dos seus produtos, proporcionando insights valiosos para a otimização do estoque e aumento do faturamento.
                    </p>
                    <p>
                        <strong>Funcionalidades Principais:</strong>
                    </p>
                    <ul>
                        <li>📂 <strong>Upload de Planilhas:</strong> Carregue seus dados em formato CSV ou XLSX.</li>
                        <li>➕ <strong>Adição Manual:</strong> Insira produtos diretamente pela interface intuitiva.</li>
                        <li>🔍 <strong>Análise Avançada:</strong> Utilize K-Means para identificar padrões e otimizar seu estoque.</li>
                        <li>📈 <strong>Visualizações Dinâmicas:</strong> Interaja com gráficos detalhados para melhor compreensão.</li>
                        <li>📄 <strong>Relatórios Personalizados:</strong> Gere PDFs profissionais com insights acionáveis.</li>
                </div>
            """, unsafe_allow_html=True)

            # Rodapé do Modal com botão
            if st.button("✨ Começar"):
                st.session_state.show_modal = False
                modal.close()

# Função para exibir o menu principal
def menu_principal():
    selected = option_menu(
        menu_title=None,
        options=["Análise Clusters", "Visualizações", "Análise Gemini"],
        icons=["bar-chart", "graph-up", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff"},
            "icon": {"color": "#27AE60", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#27AE60"},
            "nav-link-selected": {"background-color": "#d1e7dd"},
        }
    )
    return selected

# Inicializar os campos de entrada no session_state se não existirem
if 'nome_produto' not in st.session_state:
    st.session_state.nome_produto = ""
if 'preco_produto' not in st.session_state:
    st.session_state.preco_produto = 0.0
if 'quantidade_produto' not in st.session_state:
    st.session_state.quantidade_produto = 0

# Interface Principal
exibir_logo()
exibir_pop_up()
selected = menu_principal()

# Sidebar para Upload de Planilha e Adição Manual
st.sidebar.header("📥 Upload de Planilha")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV ou XLSX", type=["csv", "xlsx"])

if uploaded_file:
    df_carregado = carregar_planilha(uploaded_file)
    if df_carregado is not None:
        if 'produtos' not in st.session_state:
            st.session_state.produtos = []
        produtos_existentes = pd.DataFrame(st.session_state.produtos)
        novos_produtos = df_carregado.to_dict(orient='records')
        df_novos = pd.DataFrame(novos_produtos)
        df_combined = pd.concat([produtos_existentes, df_novos], ignore_index=True)
        st.session_state.produtos = df_combined.to_dict(orient='records')
        st.sidebar.success("Produtos carregados com sucesso!")

adicionar_produto_manual()

# Conteúdo das abas
if selected == "Análise Clusters":
    st.markdown("<div class='header-title'>📊 Análise da Curva ABC</div>", unsafe_allow_html=True)
    # Adicionando o botão dentro da aba
    if st.button("🔍 Determinar Número Ótimo de Clusters"):
        if 'produtos' in st.session_state and st.session_state.produtos:
            df = pd.DataFrame(st.session_state.produtos)
            df_abc = calcular_curva_abc(df)
            df_abc = preprocessar_dados(df_abc)
            best_k = determinar_n_clusters(df_abc)
            st.session_state.best_k = best_k
        else:
            st.error("Nenhum produto foi adicionado.")

elif selected == "Visualizações":
    st.markdown("<div class='header-title'>📈 Visualizações</div>", unsafe_allow_html=True)
    if 'produtos' in st.session_state and st.session_state.produtos:
        df = pd.DataFrame(st.session_state.produtos)
        df_abc = calcular_curva_abc(df)
        df_abc = preprocessar_dados(df_abc)
        n_clusters = st.session_state.get('best_k', 3)
        df_clusterizado, centroids_df = aplicar_kmeans(df_abc, n_clusters=n_clusters)

        # Atualizar o estado com o DataFrame clusterizado
        st.session_state.df_clusterizado = df_clusterizado

        # Verificar se 'Valor Total' existe
        if 'Valor Total' not in df_clusterizado.columns:
            st.error("A coluna 'Valor Total' está faltando no DataFrame clusterizado.")
            st.stop()

        # Definir crosstab antes de usar
        crosstab = pd.crosstab(df_clusterizado['Classe'], df_clusterizado['Cluster'])

        # Distribuição das Classes e Centróides em colunas lado a lado
        col1, col2 = st.columns(2)

        with col1:
            # Tabela de Produtos
            st.markdown("### 🗂️ Tabela de Produtos")
            st.dataframe(
                df_clusterizado[['Nome', 'Quantidade', 'Preço', 'Valor Total', '% Acumulado', 'Classe', 'Cluster']]
            )

        with col2:
            st.subheader('📊 Centróides dos Clusters')
            st.dataframe(centroids_df)

        # Gráficos adicionais
        st.markdown("### 📈 Gráficos Adicionais")

        # Criar duas linhas de gráficos com duas colunas cada para acomodar quatro gráficos
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)

        # Contar a distribuição de classes
        class_counts = df_clusterizado['Classe'].value_counts()

        # Contar a distribuição de Clusters
        class_counts2 = df_clusterizado['Cluster'].value_counts()

        with col3:
            st.subheader('📊 Classes ABC')
            fig, ax = plt.subplots(figsize=(3, 3))  # Tamanho reduzido
            ax.pie(
                class_counts,
                labels=class_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("pastel")[:len(class_counts)],
                wedgeprops={'edgecolor': 'white'}
            )
            ax.axis('equal')
            st.pyplot(fig)

        with col4:
            st.subheader('📈 Clusters')
            fig, ax = plt.subplots(figsize=(3, 3))  # Tamanho reduzido
            ax.pie(
                class_counts2,
                labels=class_counts2.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("Set2")[:len(class_counts2)],
                wedgeprops={'edgecolor': 'white'}
            )
            ax.axis('equal')
            st.pyplot(fig)

        with col5:
            st.subheader('📉 Dispersão')
            fig2, ax2 = plt.subplots(figsize=(3, 3))  # Tamanho reduzido
            sns.scatterplot(
                x='Quantidade',
                y='Preço',
                data=df_clusterizado,
                hue='Cluster',
                palette='deep',
                s=60,  # Tamanho reduzido
                edgecolor='white',
                alpha=0.7,
                ax=ax2
            )
            ax2.set_title("Dispersão de Preço x Quantidade", fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig2)

        with col6:
            st.subheader('🔥 Heatmap Classes ABC nos Clusters')
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title("Distribuição das Classes ABC nos Clusters", fontsize=14)
            st.pyplot(fig)

    else:
        st.warning("Adicione ou faça upload de produtos para visualizar as análises.")

elif selected == "Análise Gemini":
    st.markdown("<div class='header-title'>📄 Análise Gemini</div>", unsafe_allow_html=True)
    # Adicionando o botão dentro da aba
    if st.button("📤 Gerar Relatório"):
        if 'produtos' in st.session_state and st.session_state.produtos:
            df = pd.DataFrame(st.session_state.produtos)
            df_abc = calcular_curva_abc(df)
            df_abc = preprocessar_dados(df_abc)

            # Determinar o número de clusters
            n_clusters = st.session_state.get('best_k', 3)
            df_clusterizado, centroids_df = aplicar_kmeans(df_abc, n_clusters=n_clusters)

            # Verificar se 'Valor Total' existe
            if 'Valor Total' not in df_clusterizado.columns:
                st.error("A coluna 'Valor Total' está faltando no DataFrame clusterizado.")
                st.stop()

            # Contar a distribuição de classes
            class_counts = df_clusterizado['Classe'].value_counts()

            # Contar a distribuição de Clusters
            class_counts2 = df_clusterizado['Cluster'].value_counts()

            # Visualizações (já incluídas no relatório)
            crosstab = visualizar_abc_clusters(df_clusterizado)

            # Gerar análise com Gemini
            gerar_analise_gemini(df_clusterizado)

            # Gerar o PDF
            pdf = gerar_pdf(df_clusterizado, class_counts, class_counts2, st.session_state.analise_gemini)

            # Armazenar no session_state
            st.session_state['pdf'] = pdf

            st.success("Relatório gerado com sucesso! Você pode baixá-lo na barra lateral.")
        else:
            st.error("Adicione ou faça upload de produtos antes de gerar o relatório.")
    else:
        st.info("Clique no botão para gerar o relatório em PDF.")

# Chamar a função de download do PDF após as abas
baixar_pdf_sidebar()

# Rodapé
adicionar_footer()
