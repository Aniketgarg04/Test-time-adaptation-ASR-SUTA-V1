from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.data import Dataset


def load_transcripts(trans_file):
    """Load all utterances from a .trans.txt file into a dict"""
    transcripts = {}
    with open(trans_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text
    return transcripts


class LibriDataset(Dataset):
    def __init__(self, split, bucket_size, path, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # Use provided split (e.g., "test-clean", "test-other")
        if isinstance(split, str):
            split = [split]

        # Collect all audio files
        file_list = []
        for s in split:
            split_list = list(Path(os.path.join(path, s)).rglob("*.flac"))
            file_list += split_list

        # Preload all transcripts into a dictionary
        transcript_dict = {}
        for trans_file in Path(path).rglob("*.trans.txt"):
            transcript_dict.update(load_transcripts(trans_file))

        # Match each audio file with its transcript
        text = []
        valid_files = []
        for f in tqdm(file_list, desc='Read text'):
            utt_id = f.stem  # e.g. "61-70970-0000"
            transcription = transcript_dict.get(utt_id, None)
            if transcription is not None:
                text.append(transcription)
                valid_files.append(f)

        # Sort by transcript length
        self.file_list, self.text = zip(
            *[(f_name, txt) for f_name, txt in sorted(zip(valid_files, text),
                                                      reverse=not ascending,
                                                      key=lambda x: len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list) - self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size],
                        self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)
