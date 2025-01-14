# ✅ PROJECT-07

This project aims to carry out an analysis of socioeconomic data, aiming to provide a deeper understanding of the social and economic scenario in the business environment. Differences between regions, cities and countries will be explored, as well as the influence of indicators on customers' purchasing decisions and the possible impacts of changes in the scenario on corporate strategies. To achieve these objectives, exploratory analysis techniques will be applied to real, publicly available socioeconomic data. During the process, the data will be treated to solve problems identified in the cleaning step, guaranteeing the quality of the analyzes. In addition, we will adopt an interesting strategy of changing the perspective of the data, reorganizing them in an appropriate way. At the end, five relevant business questions will be answered, through the presentation of pertinent graphs and statistics.

To answer these questions, it is necessary to understand what each dataset variable represents. Next, you must load the data, identify any problems, clean the data, transform the data, organize it and then answer the questions through graphs and statistics.
 
Question 1: Does the increase in per capita GDP of a country positively affect the life expectancy of citizens at birth? What is the correlation between these two variables?
Question 2: Is there a correlation between the scale of life and the general public's awareness of corruption in business and government? What is the correlation between these two variables?
Question 3: Does increasing the scale of life have an effect on average happiness among the general public? What is the correlation between the two variables?
Question 4: Does the country with the lowest social support index have the highest perception of corruption in relation to companies and the country's government?

Keywords: R Language, Data Analysis, Statistics, dplyr, ggplot2, Data Munging, Data Exploration, Analysis Socioeconomic, Data Driven Culture, Business Problems.

# ✅ PROCESS

Correlation analysis allows us to measure the strength and direction of a linear relationship between two variables. The relationship between two variables is linear if the scatterplot between them has the straight line pattern. The relationship is positive if the slope moves upward. The relationship is negative if the slope moves down. Correlations can be perfect, moderate, weak, or unrelated. Correlation, that is, the connection between two events, does not necessarily imply a causal relationship, that is, that one of the events caused the occurrence of the other. The correlation can, however, indicate possible causes or areas for further study, i.e. the correlation can be a clue. The opposite idea, that correlation automatically proves causation, is a logical fallacy. Obviously, two events that actually have a causal relationship must also present a correlation. What constitutes the fallacy is the immediate jump to the conclusion of causality, without this being properly demonstrated.

It is necessary to develop an additional investigation depending on the different scenarios that may occur:
1. The variable x actually causes the variable y
2. The variable y can be the cause of the variable x
3. A third variable can be the cause of both variable x and variable y
4. It can be a combination of the three previous situations
5. The correlation may just be a coincidence, that is, the two events are unrelated beyond the fact that they occur at the same time. (If we're talking about a scientific study, using a large sample helps to reduce the likelihood of coincidence)

# ✅ CONCLUSION

Answer Question 1: As the Gross Domestic Product (GDP) per capita increases, an increase in life expectancy is observed. It is common to find countries with high life expectancy that also have a high GDP per capita. However, it is important to emphasize that this relationship does not imply direct causality, as it would be necessary to carry out more in-depth mathematical and statistical analyzes to confirm this with certainty. However, it is possible to identify a significant correlation between these variables, as demonstrated by the "cor.test" function. In this case, the correlation rate is approximately +85%, indicating a strong positive correlation. This information is useful for understanding the general association between GDP per capita and life expectancy, although other factors can influence this complex relationship.

![image](https://github.com/lucashomuniz/Project-7/assets/123151332/1d458130-0c84-4e14-83e5-38aa79523ad4)

Answer Question 2: There is an observed trend that, as the quality of life increases, the level of corruption decreases. Countries with greater economic prosperity generally have a better quality of life. This relationship can be attributed to several factors, such as the presence of a more educated population, greater purchasing power and broader access to information. It is important to emphasize that this association does not indicate a direct causal relationship, as further analyzes would be necessary to establish a definitive causality. However, it is clear that there is a correlation between these variables. When performing an analysis using the "cor.test" function, a correlation rate of approximately -46% is observed, which indicates a moderate negative correlation. This suggests that, in general, as the quality of life increases, the level of corruption tends to decrease. However, it is important to consider that this relationship can be influenced by several contextual and country-specific factors.

![image](https://github.com/lucashomuniz/Project-7/assets/123151332/1a158fcc-a6b8-4a53-a911-a6a1a965ebd6)

Answer Question 3: As quality of life improves, a positive impact on people's emotions and happiness is observed. There is a significant correlation between increased quality of life and the presence of more positive emotions. This relationship can be explained by several factors, such as satisfaction with important aspects of life, access to resources and opportunities, as well as emotional stability provided by favorable living conditions. However, it is important to highlight that this association does not imply a direct causal relationship, as other factors can influence a person's emotions and happiness. When performing an analysis using the "cor.test" function, it is possible to observe a correlation rate of approximately 57%, indicating a moderate positive correlation between quality of life and positive emotions. This information suggests that, in general, as the quality of life increases, there is a tendency to increase the experience of positive emotions and the feeling of happiness. However, it is important to consider that happiness is a multifaceted concept and influenced by several individual and contextual factors.

![image](https://github.com/lucashomuniz/Project-7/assets/123151332/fff9c54e-76e5-43f5-9fc2-8b191ad086a3)

Answer Question 4: There is an apparent association between the lowest level of social support in a country and a high perception of corruption indicators. Social support refers to government measures to support the population. When social support is low, theoretically, the government is collaborating less with public policies to help people. A corruption indicator resulted in a high value of 84%. This suggests that people are receiving less assistance and are considering the possibility of active government corruption. However, it is important to note that analysis of this particular country is challenging due to limited data availability. The graph generated reveals the presence of few data points, which makes it difficult to analyze the relationship between these specific data and the country in question.

When performing a correlation coefficient analysis, a value of -49% is observed, indicating a moderate negative correlation. This information suggests a possible association between lower social support and the perception of high corruption, although it is necessary to consider other variables and contextual factors that may influence this relationship. In short, the preliminary analysis indicates a possible link between low social support and the perception of corruption, but it is necessary to carry out more in-depth studies and consider other socioeconomic and political aspects for a more complete understanding of these relationships.

![image](https://github.com/lucashomuniz/Project-7/assets/123151332/13ccf588-9eb5-41f3-9ef8-21057204d661)

# ✅ DATA SOURCERS

Dataset: https://data.world/laurel/world-happiness-report-data![image](https://github.com/lucashomuniz/Project-7/assets/123151332/d514791b-82c5-43da-ae45-a373462c07a8)
