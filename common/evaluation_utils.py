import numpy as np
import torch
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

def evaluate_model(model, test_loader, class_names, device):
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                sequences, labels = batch
            else:
                sequences, labels = batch[0], batch[1]
                
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Generate confusion matrix and per-class metrics
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_accuracy = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        per_class_acc[class_name] = class_accuracy
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class_acc,
        'classification_report': classification_report(y_true, y_pred, target_names=class_names),
        'predictions': y_pred,
        'true_labels': y_true
    }

def print_evaluation_results(results, model_name, class_names):
    """
    Print formatted evaluation results
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} BASELINE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    print(f"\nCLASSIFICATION REPORT:")
    print("-" * 50)
    print(results['classification_report'])
    
    print(f"\nCONFUSION MATRIX:")
    print("-" * 50)
    print("Classes:", class_names)
    cm = np.array(results['confusion_matrix'])
    print(cm)
    
    print(f"\nPER-CLASS ACCURACY:")
    print("-" * 50)
    for class_name, accuracy in results['per_class_accuracy'].items():
        print(f"{class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("="*60)

def save_results(results, model_name, output_dir="results", seed=None):
    """
    Save evaluation results to JSON file
    """
    import os
    from .seeding import set_seeds
    import inspect
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get default seed from set_seeds function if not provided
    if seed is None:
        seed = inspect.signature(set_seeds).parameters['seed'].default
    
    # Add metadata
    results['model_name'] = model_name
    results['timestamp'] = datetime.now().isoformat()
    results['seed'] = seed
    
    filename = f"{output_dir}/{model_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")
    return filename

def compare_baselines(results_list, model_names):
    """
    Compare results from multiple baselines
    
    Args:
        results_list: List of evaluation result dictionaries
        model_names: List of model names corresponding to results
    """
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for model_name, results in zip(model_names, results_list):
        print(f"{model_name:<20} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1_score']:<12.4f}")
    
    # Find best model
    best_idx = max(range(len(results_list)), key=lambda i: results_list[i]['accuracy'])
    print(f"\nBest performing model: {model_names[best_idx]} "
          f"(Accuracy: {results_list[best_idx]['accuracy']:.4f})")
    print("="*80)