#%%
import joblib
#%%
model = joblib.load('knn_model')
print(model.predict([[-3.410692 , 0.85440 , 0.228154]]))