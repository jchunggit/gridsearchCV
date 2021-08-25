from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.linear_model import Ridge, RidgeCV
from lightgbm import LGBMRegressor
import sklearn
import lightgbm as lgbm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

####################################################################
### Estimator class
class Estimator:
    
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model,
                                       params,
                                       cv=CV_folds,
                                       **grid_kwargs,
                                      n_jobs = -1)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('Done.')
    
    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df     
      
## models, independently
## gridsearchcv per model

      
model_ls = {
  'Lasso': sklearn.linear_model.Lasso(random_state = 1234)  
}

model_rf = {
  'RandomForestRegressor': RandomForestRegressor(random_state = 1234)
}

model_xg = {
  'XGBRegressor': XGBRegressor(random_state = 1234)  
}

params_ls = {
  'Lasso': {'alpha': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10],
            'tol': [0.0001,0.001,0.01],
           'max_iter': [1000, 5000, 10000],
           'normalize': [True]}
}

params_rf = {
  'RandomForestRegressor': {'n_estimators': range(50, 500, 25), 
         'max_depth': range(50, 150, 25) }
}

params_xg = {
  'XGBRegressor': {'n_estimators': range(50, 180, 10), 
                   'max_depth': range(2, 10, 1),
                   'learning_rate': [0.1, 0.01, 0.05]}
}


####################################################################
## through all models
# gridsearchcv
models = {
  'Lasso': sklearn.linear_model.Lasso(random_state = 1234),
  'RandomForestRegressor': RandomForestRegressor(random_state = 1234),
  'XGBRegressor': XGBRegressor(random_state = 1234)  
}

params = {
  'RandomForestRegressor': {'model__n_estimators': range(50, 500, 25), 
         'model__max_features': range(50, 150, 25) },
  'Lasso': {'model__alpha': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10],
            'model__tol': [0.0001,0.001,0.01],
           'model__max_iter': [1000, 5000, 10000]},
  'XGBRegressor': {'model__n_estimators': range(50, 180, 10),
                   'model__max_depth': range(2, 10, 1),
                   'model__learning_rate': [0.1, 0.01, 0.05]}
  
}

def best_model(df):
  models = set(df['estimator'])
  for i in models:
    model_df = df[df['estimator']==i]
    model_i = model_df.mean_test_score.idxmax()
    print(i,": ", model_df.iloc[model_i]['mean_test_score'], 
          " / Params: ", model_df.iloc[model_i]['params'])