#Imports
import argparse

from datetime import datetime
from sklearn.model_selection import train_test_split

from .config import CONFIG
from .dataset import load_data_from_csv, preprocess_dataframe
from .modeling.trainer import ModelTrainer
from .modeling.mlflow_tracker import MLflowTracker

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a machine learning model.')

    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for Logistic Regression.")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="The Elastic-Net mixing parameter for Logistic Regression.")
    parser.add_argument("--class_weight", choices=['none', "balanced"], default='none', help="Class weight for Logistic Regression.")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum number of iterations for Logistic Regression.")
    parser.add_argument("--random_state", type=int, default=17, help="Random state for reproducibility.")

    return parser.parse_args()


def split_data_train_test_validate(dataframe, args_random_state):
    #Pending implementation to use the processed data from a CSV file instead of reprocessing the raw data every time
    #path_processed_data = "Proyecto/operaciones_aprendizaje_automatico/data/processed_data.csv"
    #dataframe = pd.read_csv(path_processed_data)
    
    X = dataframe.drop(columns=CONFIG.target_column)
    y = dataframe[CONFIG.target_column]

    # Split the data into train, validation and test sets using stratified sampling to maintain the proportion of classes in each set.
    Xtrain, Xtv, ytrain, ytv = train_test_split(X, y, train_size=CONFIG.train_size, stratify=y, random_state=args_random_state)

    validation_ratio = CONFIG.validation_size / (CONFIG.validation_size + CONFIG.test_size)
    
    Xval, Xtest, yval, ytest = train_test_split(Xtv, ytv, test_size=validation_ratio, stratify=ytv, random_state=args_random_state)

    print("Variables de entrada Train, Validation, Test:")
    print(Xtrain.shape, '%.1f%%' % (100.*Xtrain.shape[0]/X.shape[0]))
    print(Xval.shape, '%.1f%%' % (100.*Xval.shape[0]/X.shape[0])),
    print(Xtest.shape, '%.1f%%' % (100.*Xtest.shape[0]/X.shape[0]))

    print("\nVariables de salida Train, Validation, Test:")
    print(ytrain.shape)
    print(yval.shape)
    print(ytest.shape)

    return Xtrain, ytrain, Xval, yval, Xtest, ytest


def main():
    args = parse_arguments()

    print("Parámetros utilizados:")
    print(f"C: {args.C}")
    print(f"l1_ratio: {args.l1_ratio}")
    print(f"class_weight: {args.class_weight}")
    print(f"max_iter: {args.max_iter}")
    print(f"random_state: {args.random_state}")

    #Initialize MLflow tracker and check server connectivity before starting the training process
    mlflow_tracker = MLflowTracker(CONFIG)

    try:
        mlflow_tracker.setup_experiment()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    #Load dataframe data
    dataframe = load_data_from_csv()

    #Preprocess the dataframe data
    dataframe = preprocess_dataframe(dataframe, dropNa=True, dropDuplicates=True, dropColumns=None)

    #Split the data into train, validation and test sets
    Xtrain, ytrain, Xval, yval, Xtest, ytest = split_data_train_test_validate(dataframe, args.random_state)

    # Crear un nombre único para cada corrida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"Model_ElasticNet_{timestamp}"
    print(f"\nIniciando la corrida de MLflow con nombre: {run_name}")

    #Start an MLflow run to track the training process and log parameters and metrics
    try:
        with mlflow_tracker.start_run(run_name=run_name):
            #Train model using arguments provided by the user
            trainer = ModelTrainer(args, CONFIG)
            model = trainer.fit(Xtrain, ytrain)

            #Evaluate the model using the training set
            train_metrics = trainer.evaluate(Xtrain, ytrain)
            print(f"\nTraining metrics: {train_metrics}")

            #Evaluate the model using the validation set
            validation_metrics = trainer.evaluate(Xval, yval)
            print(f"\nValidation metrics: {validation_metrics}")

            #Evaluate the model using the test set
            test_metrics = trainer.evaluate(Xtest, ytest)
            print(f"Test metrics: {test_metrics}")

            #Hyperparameters of the model
            mlflow_tracker.log_parameter("modelo", "LogisticRegression")
            mlflow_tracker.log_parameter("C", args.C)
            mlflow_tracker.log_parameter("l1_ratio", args.l1_ratio)
            mlflow_tracker.log_parameter("class_weight", args.class_weight)
            mlflow_tracker.log_parameter("max_iter", args.max_iter)
            mlflow_tracker.log_parameter("random_state", args.random_state)

            #Metrics
            mlflow_tracker.log_metrics("train", train_metrics)
            mlflow_tracker.log_metrics("validation", validation_metrics)
            mlflow_tracker.log_metrics("test", test_metrics)

            mlflow_tracker.log_evaluation_artifacts(
                model=model,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                Xval=Xval,
                yval=yval,
                Xtest=Xtest,
                ytest=ytest
            )

            #Model registration in mlflow
            mlflow_tracker.log_model(model)

    except Exception as exc:
        print(f"ERROR during training or MLflow logging: {exc}")
        raise

if __name__ == "__main__":
    main()