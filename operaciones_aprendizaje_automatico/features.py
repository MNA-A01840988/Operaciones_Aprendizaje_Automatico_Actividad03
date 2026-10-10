

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_features():
    #Transformadores para las columnas del DataFrame, para poder generar un modelo de predicción con regresión logística.
    # NUMÉRICAS:
    numericas_power_pipeline = Pipeline( steps=[('impMediana', SimpleImputer(strategy='median')),
                                                ('powerNum', PowerTransformer(method='yeo-johnson', standardize=True))] )

    numericas_power_nombres = ['MonthlyIncome',
                            'NumCompaniesWorked',
                            'TotalWorkingYears',
                            'YearsAtCompany',
                            'YearsSinceLastPromotion']

    numericas_pipeline = Pipeline( steps = [('impMediana', SimpleImputer(strategy='median')),
                                            ('escalaNum', MinMaxScaler())] )

    numericas_pipeline_nombres = ['TrainingTimesLastYear',
                                'Age',
                                'DailyRate',
                                'DistanceFromHome',
                                'HourlyRate',
                                'MonthlyRate',
                                'PercentSalaryHike',
                                'YearsInCurrentRole',
                                'YearsWithCurrManager']

    # ORDINALES:
    catOrd_pipeline = Pipeline( steps = [('impModa', SimpleImputer(strategy='most_frequent')),
                                        ('codOrdinal', OrdinalEncoder())] )

    catOrd_pipeline_nombres = ['Education',
                            'EnvironmentSatisfaction',
                            'JobInvolvement',
                            'JobLevel',
                            'JobSatisfaction',
                            'PerformanceRating',
                            'RelationshipSatisfaction',
                            'StockOptionLevel',
                            'WorkLifeBalance']

    # BINARIAS:
    catBin_pipeline = Pipeline( steps = [('impModa', SimpleImputer(strategy='most_frequent')),
                                        ('codBinaria', OneHotEncoder(drop='if_binary',handle_unknown='ignore'))] )

    catBin_pipeline_nombres = ['Gender',
                            'OverTime']

    # NOMINALES:
    catNom_pipeline = Pipeline( steps = [('impModa', SimpleImputer(strategy='most_frequent')),
                                        ('codOneHot', OneHotEncoder(drop='first',handle_unknown='ignore'))] )

    catNom_pipeline_nombres = ['BusinessTravel',
                            'Department',
                            'EducationField',
                            'JobRole',
                            'MaritalStatus']

    columnasTransformer = ColumnTransformer( transformers = [('numPower', numericas_power_pipeline, numericas_power_nombres),
                                                            ('numpipe', numericas_pipeline, numericas_pipeline_nombres),
                                                            ('carOrd', catOrd_pipeline, catOrd_pipeline_nombres),
                                                            ('catBin', catBin_pipeline, catBin_pipeline_nombres),
                                                            ('catNom', catNom_pipeline, catNom_pipeline_nombres)],
                                            remainder='drop' )

    return columnasTransformer