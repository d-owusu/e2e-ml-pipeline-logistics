import streamlit as st
import pandas as pd
import joblib
import wandb
import os
import xgboost as xgb
import hopsworks
import yaml

with open("app_config.yaml","r") as file:
    config = yaml.safe_load(file)

hopswork_key = config["hopswork_api_key"]
wandb_key = config["wandb_api_key"]
wandb_project = config["wandb_project"]
wandb_username  = config["wandb_user"]
cts_cols=config["cts_col_names"]
cat_cols=config["cat_col_names"]
encode_columns=config["encode_column_names"]

try:
    project = hopsworks.login(api_key_value=hopswork_key)
    fs = project.get_feature_store()

    final_data_df_fg = fs.get_feature_group('final_data', version=1)
    query = final_data_df_fg.select_all()
    final_merge=query.read()
    
except Exception as e:
    st.error(f"An error occurred while fetching data from hopsworks. Please refresh the application. Error details:\n {e}")
else:    
    if not final_merge.empty:
        final_merge=final_merge.dropna()
    else:
        st.error("Empty dataframe received from hopsworks. Please check")



st.title('Truck Delay Classification')

# Let's assume you have a list of options for your checkboxes
options = ['date_filter', 'truck_id_filter', 'route_id_filter']

# Use radio button to allow the user to select only one option for filtering
selected_option = st.radio("Choose an option:", options)


if selected_option == 'date_filter':
    st.write("### Date Ranges")

    #Date range
    from_date = st.date_input("Enter start date in YYYY-MM-DD : ", value=min(final_merge['departure_date']))
    to_date = st.date_input("Enter end date in YYYY-MM-DD : ", value=max(final_merge['departure_date']))

elif selected_option == 'truck_id_filter':

    st.write("### Truck ID")
    truck_id = st.selectbox('Select truck ID: ', final_merge['truck_id'].unique())
    
elif selected_option=='route_id_filter':
    st.write("### Route ID")
    route_id = st.selectbox('Select route ID: ', final_merge['route_id'].unique())
    
if st.button('Predict'):
    try:
        flag = True

        if selected_option == 'date_filter':
            sentence = "during the chosen date range"
            filter_query = (final_merge['departure_date'] >= str(from_date)) & (final_merge['departure_date'] <= str(to_date))
        
        elif selected_option == 'truck_id_filter':
            sentence = "for the specified truck ID"
            filter_query = (final_merge['truck_id'] == truck_id)
    
        elif selected_option=='route_id_filter':
            sentence = "for the specified route ID"
            filter_query = (final_merge['route_id'] == str(route_id))
        
        else:
            st.write("Please select at least one filter")
            flag = False
    
        if flag:
            try:
                data = final_merge[filter_query] 
            except Exception as e:
                st.error(f"There is an error occurred while filtering the data with given parameters {e}")

        if data.shape[0]==0 and flag == True:
            st.error("No data was found for the selected filters. Please consider choosing different filter criteria.")
        else:
            try:
                y_test=data['delay']

                X_test=data[cts_cols+cat_cols]

                truck_data_encoder = joblib.load('truck_data_encoder.pkl')
                truck_data_scaler = joblib.load('truck_data_scaler.pkl')

                encoded_features = list(truck_data_encoder.get_feature_names_out(encode_columns))

                X_test[encoded_features] = truck_data_encoder.transform(X_test[encode_columns])
                # test
                X_test[cts_cols]  = truck_data_scaler.transform(X_test[cts_cols])

                X_test=X_test.drop(encode_columns,axis=1)
            except Exception as e:
                st.error(f"An error occurred while data preprocessing: {e}")    
            
            else:
                try:
                    os.environ['WANDB_API_KEY'] = wandb_key
                    PROJECT_NAME = wandb_project
                    USER_NAME= wandb_username
                    wandb_run = wandb.init(project=PROJECT_NAME)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                else:
                    try:
                        model_artifact = wandb_run.use_artifact(f"{USER_NAME}/{PROJECT_NAME}/XGBoost:latest", type="model")
                        model_dir = model_artifact.download()
                        wandb_run.finish()
                        model_path = os.path.join(model_dir, "xgb-truck-model.pkl")
                        model = joblib.load(model_dir + "/xgb-truck-model.pkl")
                    except Exception as e:
                        st.error(f"An Error occurred while loading the model from wandb: {e}")
                    else:
                        try:
                            dtest = xgb.DMatrix(X_test, label=y_test)
                            y_preds = model.predict(dtest)
                            X_test['Delay'] = y_preds
                            X_test['truck_id'] = data['truck_id']
                            X_test['route_id'] = data['route_id']
                            result = X_test[['truck_id','route_id','Delay']]
                        except Exception as e:
                            st.error(f"An error occurred during prediction {e}")  
                        else:      
                            if not result.empty:
                                st.write("## Truck Delay prediction "+sentence)
                                st.dataframe(result)
                            else: 
                                st.write(f"Oops!! The model failed to predict anything.")    

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")