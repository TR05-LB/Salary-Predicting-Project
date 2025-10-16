import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score, median_absolute_error

#load dataset csv
def load_dataset(file_path):
    """
    Loads a dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

#display dataset info
def kaggle_data_set_info(df):
   
    print("DataFrame Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nBasic Statistics:\n", df.describe())

def preprocess_data_for_correlation(data):
    
    """
    Preprocesses the data by one-hot encoding categorical columns and scaling numerical columns.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame with one-hot encoded categorical variables.
    """
    # Create a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Identify categorical and numerical columns
    categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
    numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
    
    print(f"Categorical columns: {list(categorical_columns)}")
    print(f"Numerical columns: {list(numerical_columns)}")
    
    # One-hot encode categorical columns
    for col in categorical_columns:
        # Get dummies and add them to the dataframe
        dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
        processed_data = pd.concat([processed_data, dummies], axis=1)
    
    # Drop original categorical columns
    processed_data = processed_data.drop(columns=categorical_columns)
    
    # Handle missing values
    processed_data = processed_data.fillna(processed_data.mean())

    return processed_data

#plot PCA
def plot_pca_with_categorical(data, n_components=2, title='PCA Plot (All Features)'):
    """
    Plots PCA including one-hot encoded categorical variables.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    n_components (int): Number of PCA components to compute.
    title (str): The title of the plot.
    """
    # Preprocess data
    processed_data = preprocess_data_for_correlation(data)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.show()
    # Print explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

def plot_correlation_heatmap_with_categorical(data, title='Correlation Heatmap (All Features)', figsize=(15, 12)):
    """
    Plots a correlation heatmap including one-hot encoded categorical variables.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    title (str): The title of the heatmap.
    figsize (tuple): Figure size for the plot.
    """
    # Preprocess data to include one-hot encoded features
    processed_data = preprocess_data_for_correlation(data)
    
    # Calculate correlation matrix
    corr = processed_data.corr()
    
    # Create the plot
    plt.figure(figsize=figsize)
    
   
    
    # Plot heatmap
    sns.heatmap(corr, 
                annot=False,  # Set to False for readability with many features
                fmt=".2f", 
                cmap='coolwarm', 
                square=True,
              cbar=True,
                cbar_kws={"shrink": .8})
    
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return processed_data
# filepath: c:\Users\tahar\Documents\AUB Courses\490\Salary-Predicting-Project\data_visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def prepare_data_for_training(data, target_column='salary_in_usd', test_size=0.2, random_state=42):
    """
    Prepares the data for machine learning by splitting into features and target.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column to predict.
    test_size (float): Proportion of dataset to include in test split.
    random_state (int): Random state for reproducibility.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, processed_data
    """
    # Preprocess data (one-hot encode categorical features)
    processed_data = preprocess_data_for_correlation(data)
    
    # Separate features and target
    if target_column in processed_data.columns:
        X = processed_data.drop(columns=[target_column, 'salary'])
        y = processed_data[target_column]
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n=== Data Preparation ===")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target variable: {target_column}")
    print(f"Number of features: {X.shape[1]}")
    print()
    return X_train, X_test, y_train, y_test, processed_data

def train_salary_prediction_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple models to predict salary and compares their performance.
    
    Parameters:
    X_train, X_test, y_train, y_test: Train/test splits
    
    Returns:
    dict: Dictionary containing trained models and their scores
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Polynomial Ridge (degree=2)': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ]),
        'Polynomial Ridge (degree=3)': Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10.0))
        ]),
        'Polynomial Lasso (degree=2)': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=1.0))
        ]),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6)
    }
    
    results = {}
    
    print(f"\n=== Training Models to Predict Salary ===")
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'predictions': y_pred_test
            }
            
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test MSE: ${test_mse:,.2f}")
            print(f"Test MAE: ${test_mae:,.2f}")
            
            # Show feature count for polynomial models
            if 'Polynomial' in name and hasattr(model, 'named_steps'):
                if hasattr(model.named_steps['poly'], 'n_output_features_'):
                    print(f"Number of polynomial features: {model.named_steps['poly'].n_output_features_}")
                
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results

def plot_salary_predictions(results, y_test):
    """
    Plots comparison of model predictions vs actual salary values.
    
    Parameters:
    results (dict): Dictionary containing model results
    y_test: True test values
    """
    fig, axes = plt.subplots(1, len(results), figsize=(18, 6))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Scatter plot of predictions vs actual
        ax.scatter(y_test, result['predictions'], alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y_test.min(), result['predictions'].min())
        max_val = max(y_test.max(), result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Salary (USD)')
        ax.set_ylabel('Predicted Salary (USD)')
        ax.set_title(f'{name}\nR² = {result["test_r2"]:.4f}, MSE = ${result["test_mse"]:,.0f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format axes to show currency
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(model, feature_names, model_name, top_n=15):
    """
    Analyzes and plots feature importance for tree-based models.
    
    Parameters:
    model: Trained model with feature_importances_ attribute
    feature_names: List of feature names
    model_name: Name of the model for the title
    top_n: Number of top features to display
    """
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for Salary Prediction ({model_name})')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{top_features.iloc[i]["importance"]:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    else:
        print(f"{model_name} doesn't have feature importances")
        return None

def plot_model_comparison_metrics(results):
    """
    Creates a bar plot comparing different model metrics.
    """
    metrics = ['test_r2', 'test_mse', 'test_mae']
    model_names = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        
        bars = axes[idx].bar(model_names, values)
        axes[idx].set_title(f'{metric.upper().replace("_", " ")}')
        axes[idx].set_ylabel('Score' if 'r2' in metric else 'USD')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if 'r2' in metric:
                label = f'{value:.4f}'
            else:
                label = f'${value:,.0f}'
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                          label, ha='center', va='bottom')
        
        # Format y-axis for currency metrics
        if metric in ['test_mse', 'test_mae']:
            axes[idx].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    plt.show()

def predict_salary_example(model, feature_names, processed_data):
    """
    Shows an example of how to predict salary for new data.
    """
    print("\n=== Example Salary Prediction ===")
    
    # Get a random sample from the data
    sample_idx = np.random.randint(0, len(processed_data))
    sample_features = processed_data.drop(['salary_in_usd', 'salary'], axis=1).iloc[sample_idx:sample_idx+1]
    actual_salary = processed_data['salary_in_usd'].iloc[sample_idx]
    
    # Make prediction
    predicted_salary = model.predict(sample_features)[0]
    
    print(f"Sample features (first 10):")
    for i, (feature, value) in enumerate(sample_features.iloc[0].items()):
        if i < 10:  # Only show first 10 features
            print(f"  {feature}: {value}")
    print("  ...")
    
    print(f"\nActual Salary: ${actual_salary:,.2f}")
    print(f"Predicted Salary: ${predicted_salary:,.2f}")
    print(f"Prediction Error: ${abs(actual_salary - predicted_salary):,.2f}")

def handle_outliers_and_preprocessing(data):
    """
    Handles outliers and improves data preprocessing for better model performance.
    """
    # Create a copy
    processed_data = data.copy()
    
    # 1. Remove extreme salary outliers using IQR method
    Q1 = processed_data['salary_in_usd'].quantile(0.25)
    Q3 = processed_data['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"Original dataset size: {len(processed_data)}")
    print(f"Salary range before outlier removal: ${processed_data['salary_in_usd'].min():,.0f} - ${processed_data['salary_in_usd'].max():,.0f}")
    
    # Remove outliers
    processed_data = processed_data[
        (processed_data['salary_in_usd'] >= lower_bound) & 
        (processed_data['salary_in_usd'] <= upper_bound)
    ]
    
    print(f"Dataset size after outlier removal: {len(processed_data)}")
    print(f"Salary range after outlier removal: ${processed_data['salary_in_usd'].min():,.0f} - ${processed_data['salary_in_usd'].max():,.0f}")
    
    return processed_data

def preprocess_data_for_correlation_improved(data):
    """
    Improved preprocessing with better handling of categorical variables.
    """
    # Handle outliers first
    processed_data = handle_outliers_and_preprocessing(data)
    
    # Identify categorical and numerical columns
    categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
    numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
    
    print(f"Categorical columns: {list(categorical_columns)}")
    print(f"Numerical columns: {list(numerical_columns)}")
    
    # One-hot encode categorical columns
    for col in categorical_columns:
        dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
        processed_data = pd.concat([processed_data, dummies], axis=1)
    
    # Drop original categorical columns
    processed_data = processed_data.drop(columns=categorical_columns)
    
    # Handle missing values
    processed_data = processed_data.fillna(processed_data.mean())
    
    return processed_data

def train_salary_prediction_models_improved(X_train, X_test, y_train, y_test):
    """
    Improved model training with better hyperparameters and regularization.
    """
    models = {
        'Linear Regression': LinearRegression(),
        
        'Ridge Regression': Ridge(alpha=100.0),
        
        'Lasso Regression': Lasso(alpha=10.0),
        
        'Polynomial Ridge (degree=2)': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=100.0))  # Higher regularization
        ]),
        
        'Polynomial Ridge (degree=2, interactions only)': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=50.0))
        ]),
        
        'Random Forest': RandomForestRegressor(
            n_estimators=300, 
            random_state=42, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt'
        ),
        
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=300, 
            random_state=42, 
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2
        )
    }
    
    results = {}
    
    print(f"\n=== Training Improved Models to Predict Salary ===")
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'predictions': y_pred_test
            }
            
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test MSE: ${test_mse:,.0f}")
            print(f"Test MAE: ${test_mae:,.0f}")
            
            # Show feature count for polynomial models
            if 'Polynomial' in name and hasattr(model, 'named_steps'):
                if hasattr(model.named_steps['poly'], 'n_output_features_'):
                    print(f"Number of polynomial features: {model.named_steps['poly'].n_output_features_}")
                
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results

def calculate_comprehensive_metrics(y_true, y_pred, model_name):
    """
    Calculates comprehensive accuracy metrics for regression models.
    
    Parameters:
    y_true: Actual values
    y_pred: Predicted values
    model_name: Name of the model
    
    Returns:
    dict: Dictionary with all metrics
    """
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Additional metrics
    explained_var = explained_variance_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    
    # Percentage-based metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    # Custom accuracy metrics
    within_10_percent = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100
    within_20_percent = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.20) * 100
    within_30_percent = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.30) * 100
    
    metrics = {
        'model_name': model_name,
        'r2_score': r2,
        'explained_variance': explained_var,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'median_ae': median_ae,
        'mape': mape,
        'within_10_percent': within_10_percent,
        'within_20_percent': within_20_percent,
        'within_30_percent': within_30_percent
    }
    
    return metrics

def print_model_accuracy_report(metrics):
    """
    Prints a comprehensive accuracy report for a model.
    """
    print(f"\n{'='*50}")
    print(f"ACCURACY REPORT: {metrics['model_name']}")
    print(f"{'='*50}")
    
    print(f"📊 GOODNESS OF FIT:")
    print(f"   R² Score:              {metrics['r2_score']:.4f}")
    print(f"   Explained Variance:    {metrics['explained_variance']:.4f}")
    
    print(f"\n💰 ERROR METRICS (in USD):")
    print(f"   MSE:                   ${metrics['mse']:,.0f}")
    print(f"   RMSE:                  ${metrics['rmse']:,.0f}")
    print(f"   MAE:                   ${metrics['mae']:,.0f}")
    print(f"   Median Absolute Error: ${metrics['median_ae']:,.0f}")
    
    print(f"\n📈 PERCENTAGE ACCURACY:")
    print(f"   MAPE:                  {metrics['mape']:.2f}%")
    print(f"   Within ±10%:           {metrics['within_10_percent']:.1f}%")
    print(f"   Within ±20%:           {metrics['within_20_percent']:.1f}%")
    print(f"   Within ±30%:           {metrics['within_30_percent']:.1f}%")
    
    # Interpretation
    print(f"\n🎯 INTERPRETATION:")
    if metrics['r2_score'] >= 0.8:
        print("   ✅ Excellent model fit")
    elif metrics['r2_score'] >= 0.6:
        print("   ✅ Good model fit")
    elif metrics['r2_score'] >= 0.4:
        print("   ⚠️  Moderate model fit")
    else:
        print("   ❌ Poor model fit")
    
    if metrics['within_20_percent'] >= 70:
        print("   ✅ High prediction accuracy")
    elif metrics['within_20_percent'] >= 50:
        print("   ⚠️  Moderate prediction accuracy")
    else:
        print("   ❌ Low prediction accuracy")

def compare_all_models_accuracy(model_results, y_test):
    """
    Compares accuracy of all models comprehensively.
    """
    all_metrics = []
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL ACCURACY COMPARISON")
    print(f"{'='*80}")
    
    for model_name, results in model_results.items():
        y_pred = results['predictions']
        metrics = calculate_comprehensive_metrics(y_test, y_pred, model_name)
        all_metrics.append(metrics)
        print_model_accuracy_report(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics)
    
    # Sort by R² score
    comparison_df = comparison_df.sort_values('r2_score', ascending=False)
    
    print(f"\n{'='*80}")
    print("QUICK COMPARISON TABLE")
    print(f"{'='*80}")
    
    print(f"{'Model':<40} {'R²':<8} {'RMSE':<12} {'MAE':<12} {'±20%':<8}")
    print("-" * 80)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['model_name']:<40} "
              f"{row['r2_score']:<8.4f} "
              f"${row['rmse']:<11,.0f} "
              f"${row['mae']:<11,.0f} "
              f"{row['within_20_percent']:<7.1f}%")
    
    return comparison_df

def plot_accuracy_comparison(comparison_df):
    """
    Creates visualizations comparing model accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² Score comparison
    axes[0,0].bar(range(len(comparison_df)), comparison_df['r2_score'])
    axes[0,0].set_title('R² Score Comparison')
    axes[0,0].set_ylabel('R² Score')
    axes[0,0].set_xticks(range(len(comparison_df)))
    axes[0,0].set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    
    # RMSE comparison
    axes[0,1].bar(range(len(comparison_df)), comparison_df['rmse'])
    axes[0,1].set_title('RMSE Comparison (Lower is Better)')
    axes[0,1].set_ylabel('RMSE (USD)')
    axes[0,1].set_xticks(range(len(comparison_df)))
    axes[0,1].set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Percentage within 20% accuracy
    axes[1,0].bar(range(len(comparison_df)), comparison_df['within_20_percent'])
    axes[1,0].set_title('Predictions Within ±20% (Higher is Better)')
    axes[1,0].set_ylabel('Percentage (%)')
    axes[1,0].set_xticks(range(len(comparison_df)))
    axes[1,0].set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    
    # MAPE comparison
    axes[1,1].bar(range(len(comparison_df)), comparison_df['mape'])
    axes[1,1].set_title('MAPE Comparison (Lower is Better)')
    axes[1,1].set_ylabel('MAPE (%)')
    axes[1,1].set_xticks(range(len(comparison_df)))
    axes[1,1].set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
def prepare_data_for_training_improved(data, target_column='salary_in_usd', test_size=0.2, random_state=42):
    """
    Improved data preparation with better preprocessing.
    """
    # Use improved preprocessing
    processed_data = preprocess_data_for_correlation_improved(data)
    
    # Separate features and target
    if target_column in processed_data.columns:
        X = processed_data.drop(columns=[target_column, 'salary'])
        y = processed_data[target_column]
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n=== Improved Data Preparation ===")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target variable: {target_column}")
    print(f"Number of features: {X.shape[1]}")
    print()
    return X_train, X_test, y_train, y_test, processed_data
    
# Add this to your existing code at the bottom:
df = load_dataset('C:\\Users\\tahar\\Desktop\\ds_salaries.csv')
df = df.drop(['salary_currency', 'Unnamed: 0'], axis=1)
kaggle_data_set_info(df)
processed_df = preprocess_data_for_correlation(df)
plot_pca_with_categorical(df)
plot_correlation_heatmap_with_categorical(df)

# === MACHINE LEARNING TRAINING ===
print("="*60)
print("TRAINING IMPROVED MACHINE LEARNING MODELS FOR SALARY PREDICTION")
print("="*60)

# Use improved data preparation
X_train, X_test, y_train, y_test, processed_data = prepare_data_for_training_improved(df, target_column='salary_in_usd')

# Train improved models
model_results = train_salary_prediction_models_improved(X_train, X_test, y_train, y_test)

# Plot predictions vs actual
plot_salary_predictions(model_results, y_test)

# Compare model metrics
plot_model_comparison_metrics(model_results)

# Analyze feature importance for tree-based models
for model_name in ['Random Forest', 'Gradient Boosting']:
    if model_name in model_results:
        model = model_results[model_name]['model']
        feature_importance_df = analyze_feature_importance(model, X_train.columns, model_name)
        print(f"\nTop 10 Most Important Features for {model_name}:")
        print(feature_importance_df.head(10)[['feature', 'importance']])

# Show example prediction
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
best_model = model_results[best_model_name]['model']
print(f"\nUsing best model ({best_model_name}) for example prediction:")
predict_salary_example(best_model, X_train.columns, processed_data)

print(f"\n=== IMPROVED MODEL SUMMARY ===")
print("Model Performance Ranking (by R² score):")
for i, (name, result) in enumerate(sorted(model_results.items(), key=lambda x: x[1]['test_r2'], reverse=True), 1):
    print(f"{i}. {name}: R² = {result['test_r2']:.4f}, MSE = ${result['test_mse']:,.0f}")

print("\n" + "="*80)
print("COMPREHENSIVE ACCURACY EVALUATION")
print("="*80)

# Evaluate all models comprehensively
comparison_df = compare_all_models_accuracy(model_results, y_test)

# Plot accuracy comparisons
plot_accuracy_comparison(comparison_df)

# Find the best model based on different criteria
print(f"\n🏆 BEST MODELS BY DIFFERENT CRITERIA:")
print(f"Best R² Score:        {comparison_df.iloc[0]['model_name']} ({comparison_df.iloc[0]['r2_score']:.4f})")
print(f"Lowest RMSE:          {comparison_df.loc[comparison_df['rmse'].idxmin()]['model_name']} (${comparison_df['rmse'].min():,.0f})")
print(f"Best ±20% Accuracy:   {comparison_df.loc[comparison_df['within_20_percent'].idxmax()]['model_name']} ({comparison_df['within_20_percent'].max():.1f}%)")
print(f"Lowest MAPE:          {comparison_df.loc[comparison_df['mape'].idxmin()]['model_name']} ({comparison_df['mape'].min():.2f}%)")
