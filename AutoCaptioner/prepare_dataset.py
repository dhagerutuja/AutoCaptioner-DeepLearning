from utils import generate_training_data

if __name__ == '__main__':
    # Generate dataset input files (including word map)
    generate_training_data(
        dataset_name='', 
        json_annotations_path='',
        images_directory='',
        captions_per_image=5,
        min_word_occurrences=5,
        output_directory='',
        max_caption_length=50,
        image_size=256  
    )
