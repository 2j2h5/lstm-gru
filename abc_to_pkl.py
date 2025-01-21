from music21 import converter
import glob
import pickle
import os
import random

data_dir = "datasets/nottingham/originals"
abc_files = glob.glob(os.path.join(data_dir, "*.abc"))

min_midi = 21
max_midi = 108

all_index_seqs = []

for abc_file in abc_files:
    print(f"Processing file: {abc_file}")
    try:
        score = converter.parse(abc_file)

        individual_songs = [s for s in score if not isinstance(s, converter.metadata.Metadata)]

        for song in individual_songs:
            song_seq = []
            for element in song.flatten().notesAndRests:
                if element.isNote:
                    song_seq.append(element.pitch.midi)
                elif element.isRest:
                    song_seq.append(0)

            song_index_seq = []
            for midi in song_seq:
                for midi in song_seq:
                    if min_midi <= midi <= max_midi:
                        song_index_seq.append([midi - min_midi])
                    else:
                        song_index_seq.append([])
            all_index_seqs.append(song_index_seq)

    except Exception as e:
        print(f"Error processing file {abc_file}: {e}")
        continue

print(f"Total songs processed: {len(all_index_seqs)}")

random.shuffle(all_index_seqs)

train_split = 0.6
valid_split = 0.2

num_songs = len(all_index_seqs)
train_end = int(train_split * num_songs)
valid_end = int((train_split + valid_split) * num_songs)

train_set = all_index_seqs[:train_end]
valid_set = all_index_seqs[train_end:valid_end]
test_set = all_index_seqs[valid_end:]

data = {
    'train': train_set,
    'valid': valid_set,
    'test': test_set
}

output_dir = "datasets/nottingham"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "nottingham.pkl"), "wb") as f:
    pickle.dump(data, f)

print(f"Data saved at {os.path.join(output_dir, 'nottingham.pkl')}")

with open(os.path.join(output_dir, "nottingham.pkl"), "rb") as f:
    loaded_data = pickle.load(f)

print(f"Train set size: {len(loaded_data['train'])}")
print(f"Valid set size: {len(loaded_data['valid'])}")
print(f"Test set size: {len(loaded_data['test'])}")