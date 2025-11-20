from prepare_data import PrepareData

# Use a small subset of TinyStories for local testing
preparer = PrepareData(
    dataset_hf="roneneldan/TinyStories",
    sample=None,  # TinyStories doesn't use sample configs like FineWeb
    output_dir="newtraining/data_tinystories",
    chunks_per_file=100  # Smaller chunks for faster testing
)

# Run preparation
preparer.prepare_dataset()
