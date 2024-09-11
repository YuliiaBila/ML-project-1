import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'Model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def iniciate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                'LinearRegression' : LinearRegression(),
                'KNeighborsRegressor' : KNeighborsRegressor(), 
                'DecisionTreeRegressor' : DecisionTreeRegressor(),
                'RandomForestRegressor' : RandomForestRegressor(), 
                'XGBRegressor' : XGBRegressor(), 
                'CatBoostRegressor' : CatBoostRegressor(verbose=False),
                'AdaBoostRegressor' : AdaBoostRegressor(),
                'GradientBoosting' : GradientBoostingRegressor()
            }

            params = {
                'DecisionTreeRegressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2']
                },
                'RandomForestRegressor': {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'GradientBoosting': {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['sqrt', 'log2', 'auto'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'LinearRegression': {},
                'KNeighborsRegressor': {
                    'n_neighbors': [5, 7, 9, 11],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['ball_tree', 'kd_tree', 'brute']
                },
                'XGBRegressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'CatBoostRegressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.01, 0.05],
                    'iterations': [30, 50, 100]
                },
                'AdaBoostRegressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    # 'loss': ['squared', 'linear', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                               models=models, param=params)
               
            best_model_score = max(model_report.values())
            best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]
            best_model = models[best_model_name]
            logging.info(f'{best_model_name}HHHHHH')     
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info('Best found model on both training and testing dataset')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
            predicted = best_model.predict(X_test)
            r2_score_result = r2_score(y_test, predicted)

            return r2_score_result


        except Exception as e:
            CustomException(e, sys)
