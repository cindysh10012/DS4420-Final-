import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# Load your CSV data (replace with actual path if different)
@st.cache_data
def load_data():
    return pd.read_csv("Apple Financial Stamt Data 24_15.csv")


# Function to create lag features
def create_lag_features(series, lags=4):
    df_lag = pd.DataFrame(series)
    for i in range(1, lags + 1):
        df_lag[f'Lag{i}'] = df_lag.iloc[:, 0].shift(i)
    return df_lag.dropna()


# Function to build a neural network with configurable layers
def build_nn_model(input_dim, hidden_layers=[4, 3]):
    model = Sequential()
    # First hidden layer
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    # Additional hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


# Function to train the model and make predictions
def train_and_predict(X_train, y_train, X_test, hidden_layers=[4, 3], epochs=100):
    model = build_nn_model(X_train.shape[1], hidden_layers)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2,
              callbacks=[early_stop], verbose=0)
    predictions = model.predict(X_test)
    return predictions.flatten()


# Function to evaluate model performance
def calculate_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mse, mae, mape



# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📌 Project Overview", "📊 Interactive Neural Network"])

# Landing Page
if page == "📌 Project Overview":

    st.header("Apple Revenue Forecasting Using Neural Networks")

    st.markdown("""
    ### Project Summary
    This project applies machine learning techniques to forecast Apple Inc.'s quarterly revenue.  
    Our goal is to leverage historical financial data to provide insights for corporate budgeting, planning, and strategy.

    We use a **Feedforward Neural Network (FNN)** to uncover nonlinear revenue trends in the time series data.

    ### Why this matters
    Financial forecasting is crucial for decision-making in businesses. Accurate revenue predictions:
    - Help allocate resources
    - Assess risks
    - Spot emerging trends

    ### Neural Network Approach
    Our neural network model:
    - Uses 4 time-lagged values as inputs (previous 4 quarters)
    - Can be configured with different numbers of hidden layers and neurons
    - Outperforms traditional linear forecasting methods
    - Handles the nonlinear patterns in Apple's revenue growth

    ### Interactive Features
    In the Interactive Neural Network section, you can:
    - Adjust the neural network architecture
    - See how different configurations affect prediction accuracy
    - View time series forecasts and performance metrics
    """)

# Neural Network Interactive Page
elif page == "Interactive Neural Network":
    st.header("Configure Your Neural Network")

    # User inputs for neural network configuration
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)

    with col1:
        num_layers = st.slider("Number of Hidden Layers", 1, 5, 2)

    # Create input for neurons in each layer
    hidden_layers = []
    for i in range(num_layers):
        with col2 if i % 2 == 1 else col1:
            neurons = st.slider(f"Neurons in Layer {i + 1}", 1, 20, 4 if i == 0 else 3)
            hidden_layers.append(neurons)

    # Additional parameters
    with col1:
        epochs = st.slider("Training Epochs", 50, 500, 100, 50)
    with col2:
        test_size = st.slider("Test Set Size (%)", 10, 30, 20)

    # Load and prepare data
    df = load_data()

    # Check if 'Quarter' and 'Revenue' columns exist
    if 'Quarter' not in df.columns or 'Revenue' not in df.columns:
        st.error("The CSV file must contain 'Quarter' and 'Revenue' columns. Please check your data.")
        st.stop()

    df['Quarter'] = pd.to_datetime(df['Quarter'])
    df = df.sort_values('Quarter')

    # Normalize revenue
    scaler = MinMaxScaler()
    df['Revenue_norm'] = scaler.fit_transform(df[['Revenue']])

    # Create lagged features
    lag_data = create_lag_features(df['Revenue_norm'], 4)

    # Split into train and test sets
    cutoff = int(len(lag_data) * (1 - test_size / 100))
    X_train = lag_data.iloc[:cutoff, 1:]  # Lag features
    y_train = lag_data.iloc[:cutoff, 0]  # Target (Revenue_norm)
    X_test = lag_data.iloc[cutoff:, 1:]
    y_test = lag_data.iloc[cutoff:, 0]

    # Train button
    if st.button("Train Neural Network"):
        with st.spinner('Training the neural network...'):
            # Train model and predict
            y_pred_norm = train_and_predict(X_train, y_train, X_test, hidden_layers, epochs)

            # Denormalize predictions
            y_test_denorm = scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
            y_pred_denorm = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

            # Calculate metrics
            mse, mae, mape = calculate_metrics(y_test_denorm, y_pred_denorm)

            # Display metrics with descriptive text
            st.subheader("Model Performance Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            metrics_col1.metric("MSE", f"{mse:.2f}")
            metrics_col1.caption("Mean Squared Error - Lower is better")

            metrics_col2.metric("MAE", f"{mae:.2f}")
            metrics_col2.caption("Mean Absolute Error (USD Millions)")

            metrics_col3.metric("MAPE", f"{mape:.2f}%")
            metrics_col3.caption("Mean Absolute Percentage Error")

            # Find the indices for training and test data correctly
            train_indices = range(4, 4 + len(X_train))
            test_indices = range(4 + len(X_train), 4 + len(X_train) + len(X_test))

            # Create visualization dataframe with proper indexing
            train_data = {
                'Quarter': df['Quarter'].iloc[train_indices].reset_index(drop=True),
                'Type': ['Training'] * len(train_indices),
                'Actual_Revenue': df['Revenue'].iloc[train_indices].reset_index(drop=True)
            }

            test_data = {
                'Quarter': df['Quarter'].iloc[test_indices].reset_index(drop=True),
                'Type': ['Test'] * len(test_indices),
                'Actual_Revenue': df['Revenue'].iloc[test_indices].reset_index(drop=True),
                'Predicted_Revenue': y_pred_denorm
            }

            # Create separate dataframes and then concatenate
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            vis_df = pd.concat([train_df, test_df], ignore_index=True)

            # Predictions are already in the test_df and now in vis_df

            # Plot with Plotly
            st.subheader("Revenue Forecast Visualization")

            fig = go.Figure()

            # Training data
            fig.add_trace(go.Scatter(
                x=vis_df[vis_df['Type'] == 'Training']['Quarter'],
                y=vis_df[vis_df['Type'] == 'Training']['Actual_Revenue'],
                mode='lines+markers',
                name='Training Revenue',
                line=dict(color='steelblue')
            ))

            # Test data - actual
            fig.add_trace(go.Scatter(
                x=vis_df[vis_df['Type'] == 'Test']['Quarter'],
                y=vis_df[vis_df['Type'] == 'Test']['Actual_Revenue'],
                mode='lines+markers',
                name='Actual Revenue',
                line=dict(color='orange')
            ))

            # Test data - predicted
            fig.add_trace(go.Scatter(
                x=vis_df[vis_df['Type'] == 'Test']['Quarter'],
                y=vis_df[vis_df['Type'] == 'Test']['Predicted_Revenue'],
                mode='lines+markers',
                name='Predicted Revenue',
                line=dict(color='darkgreen', dash='dash')
            ))

            fig.update_layout(
                title='Neural Network Revenue Forecast',
                xaxis_title='Quarter',
                yaxis_title='Revenue (USD Millions)',
                legend_title='Legend',
                template='plotly_white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Scatter plot of predicted vs actual
            st.subheader("Prediction Accuracy")

            scatter_df = pd.DataFrame({
                'Actual': y_test_denorm,
                'Predicted': y_pred_denorm
            })

            scatter_fig = px.scatter(scatter_df, x='Predicted', y='Actual',
                                     labels={'Predicted': 'Predicted Revenue', 'Actual': 'Actual Revenue'},
                                     title='Predicted vs. Actual Revenue')

            # Add the diagonal line (perfect prediction)
            min_val = min(scatter_df['Actual'].min(), scatter_df['Predicted'].min())
            max_val = max(scatter_df['Actual'].max(), scatter_df['Predicted'].max())
            scatter_fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            scatter_fig.update_layout(
                xaxis_title='Predicted Revenue (USD Millions)',
                yaxis_title='Actual Revenue (USD Millions)',
                template='plotly_white'
            )

            st.plotly_chart(scatter_fig, use_container_width=True)

            # Visualize neural network architecture
            st.subheader("Neural Network Architecture")

            # Create a visual representation of the network
            architecture_text = f"""
            ```
            Input Layer (4 features: previous 4 quarters) → """

            for i, units in enumerate(hidden_layers):
                architecture_text += f"""
            Hidden Layer {i + 1} ({units} neurons) → """

            architecture_text += """
            Output Layer (1 neuron: predicted revenue)
            ```
            """

            st.markdown(architecture_text)

            # Add explanation of results
            st.subheader("Interpretation")
            st.write(f"""
            The neural network has achieved a Mean Absolute Percentage Error (MAPE) of {mape:.2f}%, which means on average 
            the predictions are within {mape:.2f}% of the actual values. 

            For comparison:
            - MAPE < 10%: Excellent forecast accuracy
            - MAPE 10-20%: Good forecast accuracy
            - MAPE 20-30%: Reasonable forecast accuracy
            - MAPE > 30%: Inaccurate forecast

            The scatter plot shows how well the predictions align with actual values. Points closer to the red diagonal line 
            indicate more accurate predictions.
            """)

    else:
        # Show a placeholder for the visualization when the model hasn't been trained yet
        st.info("👆 Configure your neural network parameters and click 'Train Neural Network' to see the results")

        # Display sample network architecture
        st.subheader("Neural Network Architecture Preview")

        # Create a visual representation of the network
        preview_text = f"""
        ```
        Input Layer (4 features: previous 4 quarters) → """

        for i, units in enumerate(hidden_layers):
            preview_text += f"""
        Hidden Layer {i + 1} ({units} neurons) → """

        preview_text += """
        Output Layer (1 neuron: predicted revenue)
        ```
        """

        st.markdown(preview_text)

        st.markdown("""
        ### How It Works

        The neural network learns patterns from historical data to predict future revenue:

        1. **Input Layer**: Takes the revenue from the previous 4 quarters as input features
        2. **Hidden Layers**: Process the information through neurons with non-linear activation functions
        3. **Output Layer**: Produces a single value - the predicted revenue for the next quarter

        The model is trained on historical data and tested on a separate portion of data to evaluate its performance.

        ### Try Different Configurations

        Experiment with different network architectures:
        - More layers can capture more complex patterns but might overfit
        - More neurons can learn more detailed patterns
        - Different test set sizes show how well the model generalizes
        """)