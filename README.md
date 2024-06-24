# Tweedie Neural Net for CPUE-standardization
## A case study of the blue endeavour prawn in Australia's Northern Prawn Fishery


Our research focuses on standardizing catch per unit effort (CPUE) for blue endeavour prawns (Metapenaeus endeavouri) in Australia's Northern Prawn Fishery (NPF), a major and valuable prawn fishery. Despite blue endeavour prawns forming a significant portion of NPF catches, their population dynamics remain understudied. We explore the potential of Artificial Neural Networks (ANNs) for CPUE standardization, using blue endeavour prawns as a case study and comparing ANNs with generalized linear and additive models.
We introduce a novel ANN approach for CPUE standardization, incorporating two key innovations:

- An architecture inspired by the catch equation to reduce overfitting
- The use of the Tweedie distribution to handle catch uncertainty and zero values

Our model groups variables into three modules based on the catch equation, representing catchability, fishing effort, and fish density. We estimate ANN parameters by maximizing likelihood through a coordinate descent approach, alternating between optimizing Tweedie distribution parameters (power and dispersion) and standard neural net parameters.
Results indicate that these customized ANNs enhance model fitting and effectively combat overfitting. This comparison suggests a promising avenue for applying neural networks in CPUE standardization.
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

