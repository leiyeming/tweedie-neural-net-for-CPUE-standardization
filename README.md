# Tweedie Neural Net for CPUE-standardization
## A case study of the blue endeavour prawn in Australia's Northern Prawn Fishery


We perform a study on standardizing the catch per unit effort (CPUE) for the blue endeavour prawns (Metapenaeus endeavouri) caught in the Northern Prawn Fishery (NPF), one of Australia's largest and most valuable prawn fisheries. The blue endeavour prawns constitute a significant proportion of the total NPF catches, yet there have been very limited studies on its population dynamics. This study investigated the effectiveness of Artificial Neural Networks (ANNs) for CPUE standardization. We use blue endeavour prawns as a case study and conduct a comprehensive comparison with generalized linear models and generalized additive models.

In particular, we develop a novel ANN approach for CPUE standardization with two key ideas: using an architecture inspired by the catch equation to alleviate overfitting; and using the Tweedie distribution for handling the uncertainty and zero values in the catches. Specifically, we group variables into three distinct modules based on the catch equation, with each representing catchability, fishing effort, and fish density respectively. We then estimate the parameters of our ANNs by maximizing the likelihood using a coordinate descent approach, which alternates between optimizing the Tweedie distribution parameters (power and dispersion) and the standard neural net parameters. Our findings reveal that these customized ANNs improve model fitting and effectively mitigate overfitting. The comparison suggests a promising path for the application of neural networks in CPUE standardization.

### Model used:

GLMs and GAMs used in this study:
![image](https://github.com/leiyeming/tweedie-neural-net-for-CPUE-standardization/assets/47510325/25a8a639-6683-4a54-9056-7f546fad25f2)

ANN models used in this study:

![image](https://github.com/leiyeming/tweedie-neural-net-for-CPUE-standardization/assets/47510325/c8c485f1-151d-40c9-a017-7a8af1a795a7)
![image](https://github.com/leiyeming/tweedie-neural-net-for-CPUE-standardization/assets/47510325/b3a37564-f058-403c-a1d7-164c12c81189)


For more details, please refer to the code in the "code" folder.

Training algorithm:
![image](https://github.com/leiyeming/tweedie-neural-net-for-CPUE-standardization/assets/47510325/a11c2f7f-ff17-43d4-a12f-5879b3f67da4)


### Validation
To validate the methods, we performed 5-fold cross-validation (random partition) and non-overlapping training-test partition:

![image](https://github.com/leiyeming/tweedie-neural-net-for-CPUE-standardization/assets/47510325/8640f791-caaf-43d7-ad54-f87f0a073ce6)


Results for 5-fold cross-validation:
![image](https://github.com/leiyeming/tweedie-neural-net-for-CPUE-standardization/assets/47510325/265bd5f0-737f-48de-8990-eb7b5794a56c)

Results for non-overlapping training-test partition:
![image](https://github.com/leiyeming/tweedie-neural-net-for-CPUE-standardization/assets/47510325/43eeffc0-1c27-4de5-8af6-7630dafe03ca)

