
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# --- Helper Functions ---
def load_saved_model(model_name):
    try:
        model = joblib.load(f'{model_name}.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def encode_data(df):
    encoded_df = df.copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return encoded_df, encoders

def decode_predictions(predictions, true_labels, target_encoder):
    decoded_pred = target_encoder.inverse_transform(predictions)
    decoded_true = target_encoder.inverse_transform(true_labels)
    return decoded_pred, decoded_true

def animated_progress_gauge(solved, total, attempting, duration=1.5):
    percent = (solved / total) * 100 if total > 0 else 0
    steps = 30
    delay = duration / steps
    gauge_placeholder = st.empty()
    for i in range(1, steps + 1):
        current_percent = percent * i / steps
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_percent,
            number = {'suffix': "%", 'font': {'size': 36}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "#2ecc71", 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#00bcd4'},
                    {'range': [50, 80], 'color': '#ffc107'},
                    {'range': [80, 100], 'color': '#e53935'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': percent
                }
            },
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<b>{int(solved)}</b> / {int(total)}<br><span style='color:gray;font-size:0.8em'>Solved</span><br><span style='color:gray;font-size:0.8em'>{int(attempting)} Attempting</span>", 'font': {'size': 18}}
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        gauge_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(delay)
    # Show the final gauge (in case Streamlit skips the last frame)
    gauge_placeholder.plotly_chart(fig, use_container_width=True)

# --- Pages ---
def upload_page():
    st.title("🔒 Malware Detection Dashboard")
    st.write("Upload your test dataset to get predictions and analysis")
    st.sidebar.header("Model Selection")
    model_options = {
        'Android Malware': 'WeightedModels/maldroid2_model',
        'Windows Malware': 'WeightedModels/pca2_model',
        'IoMT WiFi Malware': 'WeightedModels/iomt2_model',
        'Obfuscated Malware': 'WeightedModels/obfuscated2_model'
    }
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    uploaded_file = st.file_uploader("Upload your test CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            target_column = df.columns[-1]
            encoded_df, encoders = encode_data(df)
            X_test = encoded_df.iloc[:, :-1]
            y_true = encoded_df.iloc[:, -1]
            model = load_saved_model(model_options[selected_model_name])
            if model is not None:
                y_pred = model.predict(X_test)
                decoded_pred, decoded_true = decode_predictions(y_pred, y_true, encoders[target_column])
                st.session_state.data = {
                    'decoded_pred': decoded_pred,
                    'decoded_true': decoded_true,
                    'y_pred': y_pred,
                    'y_true': y_true,
                    'encoders': encoders,
                    'target_column': target_column
                }
                st.session_state.model_name = selected_model_name
                if st.button('Analyze Results'):
                    st.session_state.page = 'results'
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Please ensure the test data format matches the training data format")

def results_page():
    st.title(f"Analysis Results - {st.session_state.model_name}")
    if st.sidebar.button('← Back to Upload'):
        st.session_state.page = 'upload'
        st.rerun()
    data = st.session_state.get('data', None)
    if data is None:
        st.error("No data available. Please upload a file first and click 'Analyze Results'.")
        return

    # Animated Progress Gauge
    solved = int((data['decoded_true'] == data['decoded_pred']).sum())
    total = len(data['decoded_true'])
    attempting = total - solved
    animated_progress_gauge(solved, total, attempting)

    # # Sample Predictions
    # st.write("### Sample Predictions")
    # sample_df = pd.DataFrame({
    #     "True Label": data['decoded_true'][:20],
    #     "Predicted Label": data['decoded_pred'][:20]
    # })
    # st.dataframe(sample_df)

    # Class Distribution
    st.write("### Class Distribution")
    class_dist = pd.DataFrame({
        'True': pd.Series(data['decoded_true']).value_counts(),
        'Predicted': pd.Series(data['decoded_pred']).value_counts()
    }).fillna(0)
    st.dataframe(class_dist)

    # Class-wise Analysis
    st.write("### Class-wise Analysis")
    unique_labels = sorted(list(set(data['decoded_true']) | set(data['decoded_pred'])))
    class_analysis = pd.DataFrame({
        'Class': unique_labels,
        'Total Samples': [sum(data['decoded_true'] == label) for label in unique_labels],
        'Correct Predictions': [sum((data['decoded_true'] == label) & (data['decoded_pred'] == label)) for label in unique_labels],
        'Accuracy': [sum((data['decoded_true'] == label) & (data['decoded_pred'] == label)) / sum(data['decoded_true'] == label) 
                   if sum(data['decoded_true'] == label) > 0 else 0 for label in unique_labels]
    })
    fig = px.bar(class_analysis,
                x='Class',
                y='Accuracy',
                title='Class-wise Prediction Accuracy',
                color='Accuracy',
                color_continuous_scale='RdYlGn')
    st.plotly_chart(fig)

    # Detailed metrics
    st.write("### Detailed Metrics")
    report = classification_report(data['decoded_true'], data['decoded_pred'], output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    st.dataframe(metrics_df)

    # Download predictions
    predictions_df = pd.DataFrame({
        'True_Label': data['decoded_true'],
        'Predicted_Label': data['decoded_pred'],
        'Correct': data['decoded_true'] == data['decoded_pred']
    })
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name=f"predictions_{st.session_state.model_name}.csv",
        mime="text/csv"
    )

    # Show encoding information
    if st.checkbox("Show Label Encoding Information"):
        st.write("### Label Encoding Information")
        for col, encoder in data['encoders'].items():
            st.write(f"\n**{col}** encoding:")
            encoding_df = pd.DataFrame({
                'Original': encoder.classes_,
                'Encoded': range(len(encoder.classes_))
            })
            st.dataframe(encoding_df)

def main():
    if st.session_state.page == 'upload':
        upload_page()
    elif st.session_state.page == 'results':
        results_page()

if __name__ == "__main__":
    main()
