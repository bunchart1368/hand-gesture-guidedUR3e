import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """Load the saved sklearn pipeline."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def extract_logistic_regression(pipeline):
    """
    Extract the logistic regression model from the pipeline.
    Returns the model and any preprocessing steps if found.
    """
    # Find the logistic regression step in the pipeline
    logistic_model = None
    preprocessor = None
    
    for step_name, step in pipeline.named_steps.items():
        if 'logistic' in step_name.lower() or hasattr(step, 'coef_'):
            logistic_model = step
        # Check for common preprocessing steps
        elif any(x in step_name.lower() for x in ['scaler', 'pca', 'polynomial', 'preprocessing']):
            preprocessor = step
    
    return logistic_model, preprocessor

def get_feature_names(pipeline, X_sample=None):
    """
    Get feature names after preprocessing.
    Optionally provide a sample of the input data to infer column names.
    """
    feature_names = []
    
    try:
        # Try to get feature names from the pipeline
        if hasattr(pipeline, 'get_feature_names_out'):
            feature_names = pipeline.get_feature_names_out()
        # For older sklearn versions
        elif hasattr(pipeline, 'get_feature_names'):
            feature_names = pipeline.get_feature_names()
        # If the pipeline has a named_transformers_ attribute (ColumnTransformer)
        elif hasattr(pipeline, 'named_transformers_'):
            for name, transformer in pipeline.named_transformers_.items():
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out())
                elif hasattr(transformer, 'get_feature_names'):
                    feature_names.extend(transformer.get_feature_names())
    except:
        pass
    
    # If we couldn't get feature names and a sample is provided
    if not feature_names and X_sample is not None:
        if isinstance(X_sample, pd.DataFrame):
            feature_names = X_sample.columns.tolist()
        else:
            # Generate generic feature names
            feature_names = [f'feature_{i}' for i in range(X_sample.shape[1])]
    
    return feature_names

def visualize_coefficients(model, feature_names, class_names=None):
    """
    Visualize the coefficients of a logistic regression model.
    
    Parameters:
    - model: A fitted logistic regression model
    - feature_names: List of feature names
    - class_names: Optional list of class names for multiclass models
    """
    coef = model.coef_
    intercept = model.intercept_
    
    # Handle binary and multiclass cases
    if coef.shape[0] == 1:  # Binary classification
        coef = coef[0]
        intercept = intercept[0]
        class_label = class_names[1] if class_names else "Class 1"
        
        # Sort coefficients for better visualization
        indices = np.argsort(coef)
        sorted_coef = coef[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(sorted_features, sorted_coef)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Logistic Regression Coefficients for {class_label}')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.tight_layout()
        plt.show()
        
        # Print the equation
        print(f"\nLogistic Regression Equation for {class_label}:")
        equation = f"log(odds) = {intercept:.4f}"
        for i, feat in enumerate(feature_names):
            equation += f" + ({coef[i]:.4f} × {feat})"
        print(equation)
        
        print("\nProbability Equation:")
        print(f"P({class_label}) = 1 / (1 + exp(-log(odds)))")
        
    else:  # Multiclass classification
        plt.figure(figsize=(12, 10))
        
        for i, (class_coef, class_int) in enumerate(zip(coef, intercept)):
            class_label = class_names[i] if class_names else f"Class {i}"
            
            # Sort coefficients
            indices = np.argsort(class_coef)
            sorted_coef = class_coef[indices]
            sorted_features = [feature_names[i] for i in indices]
            
            plt.subplot(coef.shape[0], 1, i+1)
            plt.barh(sorted_features, sorted_coef)
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.title(f'Logistic Regression Coefficients for {class_label}')
            plt.axvline(x=0, color='k', linestyle='--')
            
            # Print the equation
            print(f"\nLogistic Regression Equation for {class_label}:")
            equation = f"log(odds) = {class_int:.4f}"
            for j, feat in enumerate(feature_names):
                equation += f" + ({class_coef[j]:.4f} × {feat})"
            print(equation)
        
        plt.tight_layout()
        plt.show()
        
        print("\nProbability Equation for multiclass:")
        print("P(class_i) = exp(log_odds_i) / sum(exp(log_odds_j) for all classes j)")

def create_heatmap(model, feature_names, class_names=None):
    """Create a heatmap of coefficients for multiclass models."""
    coef = model.coef_
    
    if coef.shape[0] > 1:  # Multiclass
        plt.figure(figsize=(12, 8))
        classes = class_names if class_names else [f"Class {i}" for i in range(coef.shape[0])]
        
        sns.heatmap(coef, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                   xticklabels=feature_names, yticklabels=classes)
        plt.title("Coefficient Heatmap by Class")
        plt.tight_layout()
        plt.show()

# Main function to extract and visualize
def analyze_logistic_regression(model_path, X_sample=None, class_names=None):
    """
    Load a saved pipeline, extract the logistic regression model,
    and visualize its coefficients.
    
    Parameters:
    - model_path: Path to the saved .pkl file
    - X_sample: Optional sample of input data to help determine feature names
    - class_names: Optional list of class names for the target variable
    """
    # Load the model
    pipeline = load_model(model_path)
    
    # Extract logistic regression from pipeline
    logistic_model, preprocessor = extract_logistic_regression(pipeline)
    
    if logistic_model is None:
        print("No logistic regression model found in the pipeline!")
        return
    
    # Get feature names
    if preprocessor:
        # If we have a preprocessor and sample data, we can try to get transformed feature names
        feature_names = get_feature_names(preprocessor, X_sample)
    else:
        # Otherwise get them from the whole pipeline or the sample
        feature_names = get_feature_names(pipeline, X_sample)
    
    # Make sure we have the right number of feature names
    if len(feature_names) != logistic_model.coef_.shape[1]:
        # If not, create generic feature names
        print(f"Feature names don't match coefficients. Using generic names.")
        feature_names = [f'feature_{i}' for i in range(logistic_model.coef_.shape[1])]
    
    # Visualize coefficients
    visualize_coefficients(logistic_model, feature_names, class_names)
    
    # For multiclass, create a heatmap
    if logistic_model.coef_.shape[0] > 1:
        create_heatmap(logistic_model, feature_names, class_names)
    
    return logistic_model, feature_names

# Example usage
if __name__ == "__main__":
    # Replace with your actual model path and data
    model_path = "hand-gesture-guidedUR3e/hand-gesture-recognition/model/keypoint6mar/keypoint_classifier_lrc_pipeline.pkl"
    
    # Optional: provide a sample of your input data to help determine feature names
    # X_sample = pd.DataFrame(...)
    
    # Optional: provide class names for better visualization
    # class_names = ["Class_0", "Class_1"] # for binary classification
    # class_names = ["Class_0", "Class_1", "Class_2"] # for multiclass with 3 classes
    
    # Run the analysis
    analyze_logistic_regression(model_path)
    
    # With sample data and class names
    # analyze_logistic_regression(model_path, X_sample, class_names)