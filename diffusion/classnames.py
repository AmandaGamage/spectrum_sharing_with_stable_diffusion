class_names=[
    "ch1_empty_ch2_empty","ch1_primary_ch2_empty","ch1_secondary_ch2_empty","ch1_collision_ch2_empty","ch1_empty_ch2_primary",
    "ch1_empty_ch2_secondary","ch1_empty_ch2_secondary","ch1_primary_ch2_primary","ch1_primary_ch2_secondary","ch1_primary_ch2_collision",
    "ch1_secondary_ch2_primary","ch1_secondary_ch2_secondary", "ch1_secondary_ch2_collision","ch1_collision_ch2_primary","ch1_collision_ch2_secondary","ch1_collision_ch2_collision"
    ]

def get_classnames(source):
      
    if source == 'spectrogram':
        return [v.replace('_', ' ') for v in class_names]
    else:
        raise ValueError(f'Unknown classname source for imagenet: {source}')
