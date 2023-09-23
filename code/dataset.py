"""
dataset.py

Implementation of class for creating datasets using LibriSpeech files
and tagging corresponding metadata.

author: Dade Wood | daw1882@rit.edu
"""
# native
import dataclasses
from glob import glob
import json
import os
import random
from typing import List, Optional

# 3rd party
import pandas as pd

# local
from mixture import Mixture


@dataclasses.dataclass
class SegmentSetting:
    """
    Class containing probability for number of speakers and noise for a particular segment of
    a mixture.
    """
    position_ms: int
    duration_ms: int
    noise_prob: float
    num_speaker_probs: list


@dataclasses.dataclass
class SegmentMetadata:
    """
    All relevant metadata for a segment in a mixture. Used for tracking settings after evaluation.
    """
    position_ms: int
    duration_ms: int
    num_speakers: int
    speakers_and_sexes: List[str]
    speaker_fns: List[str]
    noise_present: bool
    noise_fn: Optional[str]


@dataclasses.dataclass
class SampleMetadata:
    """
    Metadata for a single sample in a list of SegmentMetadata.
    """
    sample_fn: str
    segments: List[SegmentMetadata]


# Assumes using LibriSpeech dataset to create a dataset, currently
class DiarizationDataset:
    def __init__(
        self,
        data_dir,
        noise_dir,
        output_dir,
        num_samples,
        file_duration_ms,
        num_unique_speakers,
        max_num_speakers,
        proportion_only_m,
        proportion_only_f,
        segment_settings: List[SegmentSetting]
    ):
        self.data_dir = data_dir
        self.noise_dir = noise_dir
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.file_duration_ms = file_duration_ms
        self.num_unique_speakers = num_unique_speakers
        self.proportion_only_m = proportion_only_m
        self.proportion_only_f = proportion_only_f
        self.max_num_speakers = max_num_speakers

        # Check if segment settings are all valid
        self.segment_settings = segment_settings
        self._validate_settings()

    def create_dataset(self):
        """
        First gather list of available speakers and their metadata, then gather all available speakers'
        corresponding audio files, and finally, create the dataset samples from gathered audio files
        and output to directory.
        """
        self._get_speakers()
        self._get_audio_files()
        self._create_samples()

    def _validate_settings(self):
        for segment_setting in self.segment_settings:
            if len(segment_setting.num_speaker_probs) != (self.max_num_speakers + 1):
                raise ValueError("Length of SegmentSetting 'num_speaker_probs' must be equal to 'max_num_speakers'.")
            if sum(segment_setting.num_speaker_probs) != 1.0:
                raise ValueError("SegmentSetting 'num_speaker_probs' must add up to 1.0")
            if 1.0 < segment_setting.noise_prob or segment_setting.noise_prob < 0.0:
                raise ValueError("SegmentSetting 'noise_prob' must be between (0.0, 1.0).")

    def _get_speakers(self):
        """
        Get a number of unique speakers from the given dataset with a 50-50 M to F split.
        """
        num_male = self.num_unique_speakers // 2
        num_female = self.num_unique_speakers - num_male
        reader_file = os.path.join(self.data_dir, 'SPEAKERS.TXT')
        speaker_meta = pd.read_csv(
            reader_file,
            names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'],
            sep='|',
            comment=';',
        )
        speaker_meta['SEX'] = speaker_meta['SEX'].str.strip()
        speaker_meta['SUBSET'] = speaker_meta['SUBSET'].str.strip()
        speaker_meta = speaker_meta[(speaker_meta.SUBSET == 'train-clean-100') | (speaker_meta.SUBSET == 'test-clean')]
        speaker_meta = speaker_meta.drop(columns=['SUBSET', 'MINUTES', 'NAME'])
        males = speaker_meta[speaker_meta["SEX"] == "M"].sample(n=num_male)
        females = speaker_meta[speaker_meta["SEX"] == "F"].sample(n=num_female)
        self.speakers = pd.concat([males, females])

    def _get_audio_files(self):
        """
        Create a dictionary of speaker IDs to their corresponding audio files as well as a list
        of noise files available for use.
        """
        speaker_files = {}
        for idx, speaker in self.speakers.iterrows():
            pattern = str(speaker['ID']) + '*.flac'
            audio_files = []
            for directory, _, _ in os.walk(self.data_dir):
                audio_files.extend(glob(os.path.join(directory, pattern)))
            speaker_files[speaker['ID']] = audio_files
        self.speaker_files = speaker_files
        self.noise_files = glob(os.path.join(self.noise_dir, '*.wav'))

    def _create_samples(self):
        """
        Generate all samples for this dataset and write the metadata to a json file.
        """
        self.samples_created = 0
        self.sample_meta = []
        num_m_samples = int(self.num_samples * self.proportion_only_m)
        num_f_samples = int(self.num_samples * self.proportion_only_f)
        num_reg_samples = self.num_samples - num_f_samples - num_m_samples

        # Create only male voice samples
        for i in range(num_m_samples):
            potential_speakers = self.speakers[self.speakers['SEX'] == 'M'].sample(
                n=self.max_num_speakers).reset_index()
            self._create_single_sample(potential_speakers)

        # Create only female voice samples
        for i in range(num_f_samples):
            potential_speakers = self.speakers[self.speakers['SEX'] == 'F'].sample(
                n=self.max_num_speakers).reset_index()
            self._create_single_sample(potential_speakers)

        # Create both sex voice samples
        for i in range(num_reg_samples):
            potential_speakers = self.speakers.sample(
                n=self.max_num_speakers).reset_index()
            self._create_single_sample(potential_speakers)

        # Write json metadata file
        json_object = json.dumps(self.sample_meta, indent=4)
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as outfile:
            outfile.write(json_object)

    def _create_single_sample(self, potential_speakers):
        """
        Generate a single sample in the dataset using the Mixture class and tracking corresponding
        metadata.
        """
        speaker_nums = {}
        for idx, speaker_row in potential_speakers.iterrows():
            speaker_nums[speaker_row['ID']] = idx
        mixture = Mixture(duration_ms=self.file_duration_ms)
        segment_metas = []
        for setting in self.segment_settings:
            speaker_fns = []
            speakers_and_sexes = []
            num_speakers = random.choices(
                population=range(len(setting.num_speaker_probs)),
                k=1,
                weights=setting.num_speaker_probs
            )[0]
            noise = random.random() < setting.noise_prob
            if num_speakers:
                curr_speakers = potential_speakers.sample(n=num_speakers)['ID'].values
                for speaker in curr_speakers:
                    sex = self.speakers[self.speakers['ID'] == speaker]['SEX'].values[0]
                    speakers_and_sexes.append(f'{speaker}_{sex}')
                    audio_fn = random.choice(self.speaker_files[speaker])
                    speaker_fns.append(audio_fn)
                    mixture.add_speaker(
                        audio_fn=audio_fn,
                        speaker_name=f"speaker{speaker_nums[speaker]}",
                        position_ms=setting.position_ms,
                        duration_ms=setting.duration_ms,
                        audio_format='flac',
                        speaker_fn=os.path.basename(audio_fn).split('.')[0]
                    )
            noise_fn = None
            if noise:
                noise_fn = random.choice(self.noise_files)
                mixture.add_noise(
                    noise_fn=noise_fn,
                    position_ms=setting.position_ms,
                    duration_ms=setting.duration_ms,
                    audio_format='wav'
                )
            segment_metas.append(
                SegmentMetadata(
                    position_ms=setting.position_ms,
                    duration_ms=setting.duration_ms,
                    num_speakers=num_speakers,
                    speakers_and_sexes=speakers_and_sexes,
                    speaker_fns=speaker_fns,
                    noise_present=noise,
                    noise_fn=noise_fn
                )
            )
        mixture.compile(fn_no_ext=f'sample_{self.samples_created}')
        mixture.export(rttm_dir=f'{self.output_dir}\\rttm', audio_dir=f'{self.output_dir}\\audio', audio_format='wav')
        self.sample_meta.append(
            dataclasses.asdict(SampleMetadata(sample_fn=f'sample_{self.samples_created}', segments=segment_metas))
        )
        self.samples_created += 1


if __name__ == "__main__":
    # NOTE THAT DIRECTORY PATHS MUST BE CHANGED FOR YOUR SYSTEM SETUP

    # SAMPLE DATASET FOR SUBMISSION
    settings = [
        SegmentSetting(position_ms=0, duration_ms=2500, noise_prob=0.10, num_speaker_probs=[0.0, 0.50, 0.50]),
        SegmentSetting(position_ms=2500, duration_ms=2500, noise_prob=0.10, num_speaker_probs=[0.15, 0.55, 0.30]),
    ]
    d = DiarizationDataset(
        data_dir='.\\LibriSpeech',
        noise_dir='.\\musan\\noise\\free-sound',
        output_dir='.\\sample_dataset',
        num_samples=10,
        file_duration_ms=10_000,
        num_unique_speakers=5,
        max_num_speakers=2,
        proportion_only_m=0.1,
        proportion_only_f=0.1,
        segment_settings=settings
    )
    d.create_dataset()

    # DATASET A
    settings = [
        SegmentSetting(position_ms=0, duration_ms=2500, noise_prob=0.10, num_speaker_probs=[0.0, 0.50, 0.50]),
        SegmentSetting(position_ms=2500, duration_ms=2500, noise_prob=0.10, num_speaker_probs=[0.15, 0.55, 0.30]),
    ]

    d = DiarizationDataset(
        data_dir='.\\LibriSpeech',
        noise_dir='.\\musan\\noise\\free-sound',
        output_dir='.\\dataset_a',
        num_samples=100,
        file_duration_ms=10_000,
        num_unique_speakers=10,
        max_num_speakers=2,
        proportion_only_m=0.1,
        proportion_only_f=0.1,
        segment_settings=settings
    )
    d.create_dataset()

    # DATASET B
    settings = [
        SegmentSetting(position_ms=0, duration_ms=2500, noise_prob=0.10, num_speaker_probs=[0.0, 0.50, 0.50, 0.0]),
        SegmentSetting(position_ms=2500, duration_ms=2500, noise_prob=0.10, num_speaker_probs=[0.5, 0.75, 0.15, 0.05]),
    ]

    d = DiarizationDataset(
        data_dir='.\\LibriSpeech',
        noise_dir='.\\musan\\noise\\free-sound',
        output_dir='.\\dataset_b',
        num_samples=100,
        file_duration_ms=10_000,
        num_unique_speakers=10,
        max_num_speakers=3,
        proportion_only_m=0.1,
        proportion_only_f=0.1,
        segment_settings=settings
    )
    d.create_dataset()

    # DATASET C
    settings = [
        SegmentSetting(position_ms=0, duration_ms=2500, noise_prob=0.05, num_speaker_probs=[0.10, 0.55, 0.20, 0.10, 0.05]),
        SegmentSetting(position_ms=2500, duration_ms=2500, noise_prob=0.05, num_speaker_probs=[0.10, 0.40, 0.25, 0.15, 0.10]),
        SegmentSetting(position_ms=5000, duration_ms=2500, noise_prob=0.05, num_speaker_probs=[0.10, 0.75, 0.05, 0.05, 0.05]),
        SegmentSetting(position_ms=7500, duration_ms=2500, noise_prob=0.05, num_speaker_probs=[0.10, 0.40, 0.25, 0.15, 0.10]),
    ]

    d = DiarizationDataset(
        data_dir='.\\LibriSpeech',
        noise_dir='.\\musan\\noise\\free-sound',
        output_dir='.\\dataset_c',
        num_samples=100,
        file_duration_ms=10_000,
        num_unique_speakers=10,
        max_num_speakers=4,
        proportion_only_m=0.1,
        proportion_only_f=0.1,
        segment_settings=settings
    )
    d.create_dataset()

    # DATASET D
    settings = [
        SegmentSetting(position_ms=0, duration_ms=2500, noise_prob=0.20, num_speaker_probs=[0.05, 0.60, 0.20, 0.10, 0.05]),
        SegmentSetting(position_ms=2500, duration_ms=2500, noise_prob=0.20, num_speaker_probs=[0.50, 0.30, 0.15, 0.04, 0.01]),
        SegmentSetting(position_ms=5000, duration_ms=2500, noise_prob=0.20, num_speaker_probs=[0.50, 0.15, 0.30, 0.04, 0.01]),
        SegmentSetting(position_ms=7500, duration_ms=2500, noise_prob=0.20, num_speaker_probs=[0.05, 0.75, 0.10, 0.05, 0.05]),
    ]

    d = DiarizationDataset(
        data_dir='.\\LibriSpeech',
        noise_dir='.\\musan\\noise\\free-sound',
        output_dir='.\\dataset_d',
        num_samples=100,
        file_duration_ms=10_000,
        num_unique_speakers=10,
        max_num_speakers=4,
        proportion_only_m=0.1,
        proportion_only_f=0.1,
        segment_settings=settings
    )
    d.create_dataset()
