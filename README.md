# ✅ PROJECT-12

Nesse projeto vamos realizr um passo a passo completo do processo de construção, treinamento, avaliação e seleção de modelos para Regressão. Todo o processo será demonstrado de uma ponta a outra, desde a definição do problemade negócio, até a interpretação do modelo e entrega do resultado ao tomador de decisão. Uma empresa de e-commerce comercializa produtos através de seu website e através de sua aplicação para dispositivos móveis. Para efetuar uma compra, um cliente realiza um cadastro no portal (usando website ou app). Cada vez que o cliente realiza o login, o sistema registra o tempo que ele fica logado. Para cada cliente a empresa mantém o registro de vendas com o total gasto por mês. Com isso, a empresa gostaria de aumentar as vendas, mas o orçamento permite investir somente no website ou na app neste momento. Dessa forma, o objetivo é melhorar a experiência do cliente durante a navegação no sistema, aumetando o tempo logado, aumentando o engajamento e consequentemente, aumentando as vendas. Os dados deste projeto são fictícios, entretanto representam dados reais para empresas de e-commerce. Os dados rerepresentam um mês de operação do portal. O título de cada coluna no conjunto de dados é auto-explicativo.

Keywords: R Language, Microsoft PowerBI, Data Visualization, Data Analysis, Data Munging, Statistics Model, Sales Forecast Accuracy, Exploratory Analysis


# ✅ PROCESS

A Análise Exploratória é feita logo no início, após o carregamento dos dados. Nessa fase é feita toda o processo de limpeza (valores duplicados e valores ausentes), bem como possiveis transformações pontuais. O principal foco é realizar o entendimento do dataframe, com foco na visualização dos tipos das variáveis númerias e categóricas, bem como na visualização das suas distribuições e principalmente o tratamento de outliers a partir do boxplot, tabela de describe e contagem de frequência. Na Análise Exploratória não se pode ter linhas duplicadas nem colunas (variáveis) duplicadas, pelo fato de que caso isso ocorra teriamos uma dupicidade, o que teoricamente iria fazer com que o modelo desenvolvido fosse tendecioso, pois estariamos reforçando uma informação. O objetivo é ter um modelo generalizável.

A partir do gráfico de dispersão baseado na tabela de correlação podemos verificar a interação entre as variáveis. Com o aumento do tempo logado na app é nítido que resulta em um aumento do valor total gasto (correlação positiva moderada). Em um projeto de Machine Learning de Regressão, o ideal é que as variáveis preditoras tenham uma boa correlação com a variável alvo. Entretanto, não é o ideal possuir uma alta correlação entre as variáveis preditoras, pois isto pode levar a um problema de multicolinearidade.





A etapa seguinte é o processo de Engenharia de Atributos, na qual é desenvolvido um tipo de transformação mais profunda (caso necessário) bem como uma possível criação e alteração das variáveis. Uma opção dentre dess fase é fazer o Feature Selection, com o intuito de obter as melhores variáveis para seguir com o processo de Machine Learning. Por fim, uma das técnicas mais importantes nessa fase é a criacão da Tabela de Correlação, na qual vai ser possível identificar possíveis níveis (positivo ou negativo) de relacionamento entre as variáveis, pricipalmente analisar possíveis indícios de multicolinearidade entre variáveis.

A proxima etapa é o pré-processamento, onde é feito as alterações de variáveis que tenham ainda texto para número, bem como é feito a organização de todo o modelo de machine learning, escolha do algoritmo principal, label enconding, normalização, padronizaçào e scaling. Uma outra técnica bastante utilizada durante essa etapa é de dividr os dados do dataframe em dados de treino e dados de teste. Isso é importante pelo fato de que o modelo de machine learning é treinado (criado) com os dados de treino e em seguida é testadado (avaliado) com os dados de teste. Uma vez que o modelo esteja treinado, eu não posso apresentar a ele os mesmos dados utilizados em treinamento, porque ele já conhece esses dados. Para avaliar a performace do modelo, eu tenho que utilizar novos dados que eu obviamente já conheco o resultado

# ✅ CONCLUSION

# ✅ DATA SOURCES
