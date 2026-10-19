# Classificação de e-mails SPAM com MLP

Atividade prática da disciplina de **Redes Neurais e Deep Learning**, da pós-graduação da **Universidade Tecnológica Federal do Paraná (UTFPR)**. O objetivo é aplicar os conceitos estudados de redes neurais artificiais na implementação de um classificador binário de e-mails, distinguindo mensagens de spam de mensagens legítimas.

> **Contexto acadêmico:** este repositório registra uma atividade de aprendizado e experimentação. Os resultados são referentes à base utilizada e não representam, por si só, um sistema pronto para uso em produção.

## Objetivos

- Explorar e compreender uma base de dados de classificação binária.
- Investigar diferenças entre mensagens classificadas como spam e não spam.
- Implementar um **Multilayer Perceptron (MLP)** usando Python e scikit-learn.
- Avaliar o impacto da padronização e de diferentes configurações de arquitetura e taxa de aprendizado.
- Aplicar uma metodologia de validação que reduza o risco de vazamento de dados causado por registros duplicados.
- Interpretar métricas de classificação, discutir limitações e comparar o modelo com um baseline simples.

## Dataset

O projeto utiliza a base **Spambase**, também conhecida como *Spambase / SPAM E-mail Database*. Ela contém **4.601 mensagens**, representadas por **57 atributos numéricos preditores** e uma variável-alvo binária:

- **48 atributos `word_freq_*`:** frequências de palavras específicas no e-mail.
- **6 atributos `char_freq_*`:** frequências de caracteres específicos.
- **3 atributos `capital_run_length_*`:** estatísticas sobre sequências de letras maiúsculas.
- **`class`:** classe-alvo, em que `1` representa spam e `0` representa mensagem não spam.

A distribuição da base é de **1.813 mensagens de spam (39,4%)** e **2.788 mensagens não spam (60,6%)**. Não foram identificados valores ausentes na análise inicial.

A escolha da base considerou três fatores: os atributos já são numéricos, o volume de dados é relativamente pequeno e permite iterações rápidas em um computador pessoal, e o problema de filtragem de spam é um caso de aplicação de aprendizado de máquina que desperta interesse. Como os atributos são frequências e estatísticas extraídas previamente, o projeto não trabalha diretamente com o texto bruto dos e-mails.

## Metodologia

### Exploração e preparação

A análise exploratória examina a distribuição das classes, a escala e a dispersão dos atributos e as diferenças entre suas médias por classe. Colunas auxiliares que funcionam apenas como índices são excluídas dos preditores, evitando que sejam usadas como informação explicativa pelo modelo.

A análise também identifica combinações de atributos duplicadas. Para reduzir a possibilidade de mensagens idênticas ou com os mesmos preditores aparecerem simultaneamente nos conjuntos de treino e teste, os registros são agrupados por combinação de atributos durante a divisão dos dados e a validação cruzada.

### Modelo e seleção de hiperparâmetros

O classificador é implementado com `MLPClassifier`, da biblioteca scikit-learn, em um `Pipeline` que permite comparar a padronização dos atributos com a ausência dessa transformação.

Configuração selecionada no experimento:

- **Arquitetura:** uma camada oculta com 32 neurônios (`hidden_layer_sizes=(32,)`).
- **Função de ativação:** ReLU.
- **Otimizador:** Adam.
- **Taxa de aprendizado inicial:** `0.01`.
- **Número máximo de iterações:** `500`.
- **Parada antecipada:** habilitada (`early_stopping=True`).
- **Reprodutibilidade:** `random_state=42`.
- **Pré-processamento:** `StandardScaler`.

A busca de hiperparâmetros compara três arquiteturas (`(16,)`, `(32,)` e `(64, 32)`), duas taxas de aprendizado (`0.001` e `0.01`) e duas opções de pré-processamento (com e sem `StandardScaler`). A seleção usa `GridSearchCV` com validação cruzada estratificada e agrupada em três partes, usando o **F1-score** como métrica de seleção.

### Avaliação

Os dados são separados em treino e teste com divisão estratificada e agrupada, preservando aproximadamente a proporção entre as classes e evitando que grupos de preditores duplicados sejam compartilhados entre os conjuntos. A avaliação final é realizada no conjunto de teste, separado da busca de hiperparâmetros.

As métricas reportadas são acurácia, precisão, revocação (*recall*), F1-score e ROC-AUC. Também é utilizado um `DummyClassifier` com estratégia `most_frequent` como baseline para contextualizar o desempenho do MLP.

## Resultados

Resultados obtidos na avaliação final do conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,9400 (94,00%) |
| Precisão para spam | 0,9197 (91,97%) |
| Recall para spam | 0,9288 (92,88%) |
| F1-score para spam | 0,9242 |
| ROC-AUC | 0,9833 |
| F1 médio da melhor configuração na validação cruzada | 0,9115 |

O baseline que sempre prevê a classe majoritária alcança acurácia de aproximadamente **60,63%** e F1-score igual a zero para a classe spam. O MLP apresenta desempenho substancialmente superior nesse experimento.

A precisão de 91,97% indica que parte das mensagens sinalizadas como spam é, na realidade, legítima. O recall de 92,88% indica que parte das mensagens de spam não é identificada. Essa distinção importa porque falsos positivos podem ocultar mensagens legítimas, enquanto falsos negativos permitem que spam chegue à caixa de entrada. A ROC-AUC de 0,9833 descreve a capacidade de discriminação entre classes ao variar o limiar de decisão; **não** significa que o modelo tenha 98,33% de acurácia.

A arquitetura com 32 neurônios na camada oculta foi escolhida por apresentar o melhor F1 médio na validação cruzada entre as configurações testadas. A arquitetura maior `(64, 32)` teve resultado próximo, mas não trouxe ganho relevante que justificasse a complexidade adicional.

## Limitações

- **Base histórica e específica:** os padrões de vocabulário e de composição das mensagens podem não representar e-mails atuais, novos tipos de spam ou outros idiomas e contextos.
- **Atributos pré-extraídos:** o modelo não processa o texto bruto, cabeçalhos, URLs, anexos, reputação do remetente ou sinais de autenticação.
- **Interpretabilidade:** embora as métricas sejam boas, é difícil explicar individualmente por que uma rede MLP classificou determinada mensagem como spam.
- **Custo dos erros:** precisão e recall refletem um compromisso entre deixar spam passar e bloquear mensagens legítimas. O limiar ideal dependeria do contexto de uso.
- **Duplicatas:** o agrupamento na divisão externa e na validação cruzada reduz vazamento por duplicatas, mas a validação interna usada por `early_stopping=True` no `MLPClassifier` é feita sem respeitar esses grupos. Isso pode influenciar o critério de parada e deve ser considerado ao interpretar os resultados.
- **Validação limitada:** foi utilizado um único conjunto de teste e uma configuração específica de validação cruzada. Testes em bases externas e em dados mais recentes seriam necessários para avaliar a generalização.
- **Não é um sistema de produção:** uma aplicação real exigiria monitoramento, atualização de dados, análise de falsos positivos, controles de segurança e integração com outros sinais.

## Uso de inteligência artificial

Ferramentas de inteligência artificial foram utilizadas como apoio à elaboração, revisão e melhoria dos códigos, com o objetivo de acelerar as iterações, explorar implementações mais robustas e evitar que a experiência ainda em desenvolvimento com programação se tornasse um gargalo para a aplicação dos conceitos teóricos estudados.

Esse apoio não substitui a compreensão das decisões metodológicas, dos hiperparâmetros ou das métricas. A exploração dos dados, a definição do experimento e a interpretação crítica dos resultados são componentes essenciais do trabalho.

## Tecnologias

- Python
- pandas e NumPy para manipulação de dados
- scikit-learn para pré-processamento, treinamento, validação cruzada e métricas
- Matplotlib e Seaborn para visualizações
- Jupyter Notebook para experimentação

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
```

Substitua `SEU-USUARIO` e `SEU-REPOSITORIO` pelos dados do repositório criado no GitHub.

### 2. Criar um ambiente virtual

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Abrir o notebook

```bash
jupyter notebook
```

Abra o notebook principal do projeto pela interface do Jupyter e execute as células na ordem apresentada. Verifique se o arquivo `database.csv` está no caminho esperado pelo notebook. Se o nome ou a localização do notebook/dataset for diferente no repositório, ajuste o comando ou o caminho correspondente.

## Estrutura sugerida

A estrutura exata depende dos arquivos adicionados ao repositório. Uma organização simples para esta atividade seria:

```text
.
├── database.arff                 # Base utilizada no experimento
├── notebook.ipynb               # Análise exploratória e implementação do MLP
├── relatorio_final_mlp_spam.md  # Relatório da atividade
└── README.md                    # Visão geral e instruções
```

## Referências

- **Spambase / UCI Machine Learning Repository:** https://archive.ics.uci.edu/dataset/94/spambase
- **scikit-learn, `MLPClassifier`:** https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- **scikit-learn, `GridSearchCV`:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- **scikit-learn, `StratifiedGroupKFold`:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

---

Atividade acadêmica desenvolvida no contexto da pós-graduação da **UTFPR**, com finalidade didática.
