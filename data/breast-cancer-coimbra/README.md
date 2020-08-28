# Breast Cancer Coimbra Dataset

This dataset serves as a simple first toy, since other data sets used in the paper could not be published.

## Information about the Dataset

There are 10 predictors, all quantitative, and a binary dependent variable, indicating the presence or absence of breast cancer. 
The predictors are anthropometric data and parameters which can be gathered in routine blood analysis. 
Prediction models based on these predictors, if accurate, can potentially be used as a biomarker of breast cancer.

The dataset was taken from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra 

Corresponding Paper (from 2018): https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3877-1 

### Dataset Splits: 

There is only one csv-file for training, testing and validation, i.e. the splits are done manually within the data reader.

## Attribute Information in Files

#### Quantitative Attributes (9 Input Variables): 
1. Age (years) 
2. BMI (kg/m2) 
3. Glucose (mg/dL) 
4. Insulin (µU/mL) 
5. HOMA 
6. Leptin (ng/mL) 
7. Adiponectin (µg/mL) 
8. Resistin (ng/mL) 
9. MCP-1(pg/dL) 

#### Labels (1 Output Variable): 
1 = Healthy controls (no cancer)
2 = Patients (cancer)