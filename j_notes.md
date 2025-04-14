# Note on Path Handling:
Prefer using os.path.dirname(os.path.abspath(__file__)) with os.path.join() 
for file paths in Python scripts. This makes scripts location-independent and 
more portable, as they'll work correctly regardless of the current working directory.

# Important RecBole Configuration Note:
When using run_recbole(), the config_dict parameter overrides any settings specified in the YAML config files.
This means you should either:
1. Use only YAML config files (config_file_list)
2. Use only config_dict

# RecBole Configuration Files:
The framework uses a hierarchical configuration system:
1. overall.yaml: Global settings (environment, training defaults, evaluation)
2. dataset/*.yaml: Dataset-specific settings (data loading, preprocessing)
3. model/*.yaml: Model-specific settings (architecture, hyperparameters)

Configuration precedence:
- Model-specific YAML overrides dataset YAML
- Dataset YAML overrides overall YAML
- config_dict (if used) overrides all YAML settings

# Dual YAML Requirement:
When running a model, you typically need two YAML files:
1. Dataset YAML (e.g., ml-100k-wiki.yaml):
   - Defines data loading and preprocessing
   - Specifies field names and types
   - Sets up data splitting and filtering
   - Configures embedding loading if needed

2. Model YAML (e.g., DESSBERT4Rec.yaml):
   - Defines model architecture parameters
   - Sets training hyperparameters
   - Configures model-specific settings
   - Can override training parameters from overall.yaml

Example usage in run_recbole():
```python
run_recbole(
    model="DESSBERT4Rec",
    dataset="ml-100k-wiki",
    config_file_list=[
        "recbole/properties/dataset/ml-100k-wiki.yaml",
        "recbole/properties/model/DESSBERT4Rec.yaml"
    ],
)
```

# Work logg
BERT4Rec on MovieLens-100k - Quick Test

Successfully set up and ran BERT4Rec on ML-100k dataset:
- Created minimal_bert4rec.py with basic configuration
- Used atomic file format (ml-100k.inter) from ./dataset
- Split data 80/10/10 (train/val/test)
- Ran for 2 epochs with CE loss
- Evaluated using standard metrics (Recall, MRR, NDCG)

✓ Everything worked properly with default hyperparameters 

DESSBERT4Rec Setup - Quick Test

Steps taken to set up DESSBERT4Rec:
- Created DESSBERT4Rec.py by by copying "BERT4Rec" which extends SequentialRecommender
- Added DESSBERT4Rec.yaml config file with same parameters as BERT4Rec
- Created minimal_dessbert4rec.py for quick testing
- Used run_recbole() from quick_start module for execution
- Configured with same hyperparameters as BERT4Rec for initial testing

✓ Basic setup complete, ready for model modifications 

DESSBERT4Rec with Wikipedia Embeddings - Implementation

Key RecBole Documentation References:
- docs/user_guide/data/data_format.md: Dataset format specifications
- docs/user_guide/data/dataset_attributes.md: Dataset attributes and field types
- docs/user_guide/data/dataset_loading.md: Dataset loading and preprocessing
- docs/user_guide/data/dataset_combination.md: Combining multiple data sources
- docs/user_guide/data/dataset_attributes.md#preload-weight: Preload weight configuration
- docs/user_guide/data/dataset_attributes.md#normalize-field: Field normalization settings

Steps taken to integrate Wikipedia embeddings:
1. Data Preparation:
   - Created links.tsv to map item_ids to movie titles from .item file
   - Used movie_embeddings.pkl (dict with movie_title -> embeddings)
   - Created create_emb_file.py to merge and format embeddings
   - Generated ml-100k-wiki.emb with format:
     * iid:token (using iid instead of item_id to avoid conflicts)
     * wiki_embedding:float_seq (384-dim embeddings)
   - Verified embedding dimensions with check_embedding_dim.py

2. RecBole Dataset Setup:
   - Created ml-100k-wiki directory in dataset/
   - Copied original ml-100k files:
     * ml-100k-wiki.user (user information)
     * ml-100k-wiki.item (item information with movie titles)
     * ml-100k-wiki.inter (interaction data)
   - Created ml-100k-wiki.emb for embeddings
   - Created ml-100k-wiki.yaml for dataset configuration

3. Configuration Updates (ml-100k-wiki.yaml):
   - Added wiki_embedding to item_fields
   - Set preload_weight to map iid -> wiki_embedding
   - Added wiki_embedding to normalize_field
   - Configured field_separator as tab for proper parsing
   - Set preload_weight_dim to 384 to match embedding size
   - Used iid instead of item_id to avoid conflicts with original dataset

4. Model Integration:
   - Created minimal_dessbert4rec.py for testing
   - Used run_recbole() with ml-100k-wiki dataset
   - Successfully loaded and used Wikipedia embeddings in training
   - Model configuration:
     * Uses DESSBERT4Rec architecture
     * Loads embeddings as preload_weights
     * Normalizes embeddings during preprocessing
     * Maintains original interaction data structure

✓ DESSBERT4Rec now running with Wikipedia embeddings
✓ Model successfully completes training epochs
✓ Ready for performance evaluation and potential hyperparameter tuning

Key RecBole Requirements Met:
1. File Naming Convention:
   - All files follow dataset_name.suffix format
   - .emb file contains preload weights
   - Consistent naming across all dataset files

2. Field Formatting:
   - Used tab separator for complex fields
   - Proper field type specifications (token, float_seq)
   - Correct header format with field types

3. Preload Weight Setup:
   - Correct mapping in preload_weight section
   - Proper dimension specification
   - Normalization configuration

4. Dataset Structure:
   - Maintained original interaction format
   - Added embedding information without disrupting base structure
   - Proper field separation and formatting


### My own summary: at this point I believe we have a "BERT4Rec" copy that trains towards frozen-predefined embeddings using cross entropy loss. Next we need to change the loss calculation to use DESS. "FORK" This state to keep as comparison. 

# DESSBERT4Rec Implementation - DESS Loss Integration

## What I Learned from This Chat

1. RecBole has a dedicated module for loss functions in `recbole/model/loss.py`:
   - Contains common loss functions like BPRLoss, RegLoss, EmbLoss
   - Already includes a DESSLoss implementation that we can use
   - DESSLoss takes beta and alpha parameters to control the loss behavior

2. BERT4Rec Architecture Understanding:
   - The core architecture consists of transformer layers and output processing
   - The loss calculation happens in the `calculate_loss` method, not in `forward`
   - The model projects hidden vectors to item dictionary size using matrix multiplication
   - For DESSBERT4Rec, we need to project to embedding dimension instead

3. Implementation Strategy:
   - Only need to modify the `calculate_loss` method to use DESS loss
   - Add an embedding projection layer to convert from hidden_size to 2*embedding_dim
   - Use the DESSLoss class from recbole.model.loss
   - Update predict and full_sort_predict methods to work with embedding predictions

## Implementation Details

1. Added DESS Loss Support:
   - Imported DESSLoss from recbole.model.loss
   - Added beta and alpha parameters to __init__ method
   - Created embedding_projection layer in __init__ method
   - Implemented DESS loss calculation in calculate_loss method

2. Modified Prediction Methods:
   - Updated predict method to use embedding projection
   - Updated full_sort_predict method to use embedding projection
   - Both methods now extract the mu part of the prediction for similarity calculation

3. Key Changes:
   - Changed from projecting to dictionary size to projecting to embedding dimension
   - Replaced CE loss with DESS loss for embedding prediction
   - Maintained the same masking and sequence processing logic

## Next Steps

1. Test the implementation with the ml-100k-wiki dataset
2. Compare performance with the original BERT4Rec implementation
3. Fine-tune hyperparameters (beta, alpha) for optimal performance
4. Consider additional improvements to the architecture 