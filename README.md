# PersonalizedFLEducation
Assessing Academic Performance: Personalized Federated Learning for Identifying Overperforming and Underperforming Schools and Individual-Level Implications

Research conducted in Computer Science Department @ Florida State University

Young Scholars Program: https://ysp.osta.fsu.edu

### Model Details/Structure
##### Dataset: Florida Department of Education - School Grades 2022 Dataset

##### Client Model:
- Type: Deep Neural Network (DNN)
- Layers: 1 input, 4 hidden, 1 output | 16, 32, 64, 32, 16, 1
- Inputs: 16 features (e.g. school grade, % minority/economically disadvantaged, ELA/science Achievement, school type, etc)
- Output: Mathematics Achievement (value from 0-100, used as proxy for school performance)
- Loss: Mean Squared Error, Optimizer: Adam, Learning Rate: 0.001
- Local Epochs: 3, Batch Size: 128


##### Global Model:
- Type: Personalized Federated Learning
- Aggregating Epochs: 100, Batch Size: 128
- Model Weightage: 70% global, 30% local

### Model Results
##### Coefficient of Determination (r^2) Convergence
- Personalized FL: ~0.86
- FL: ~0.86
- Centralized: ~0.89


##### Mean Absolute Error (MAE) Convergence
- Personalized FL: ~5
- FL: ~5
- Centralized: ~4.5
