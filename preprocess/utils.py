import json
import os
import sys
import time

import numpy as np


BUCKET_SIZE = 1000000


class Sample(object):
    def __init__(self, sample):
        self.chrom = sample["chrom"]
        self.start = sample["start"]
        self.stop = sample["stop"]


class FilenameIterator(object):
    def __init__(self, prefix, batch_size):
        self.prefix = prefix
        self.batch_size = batch_size

        self.initialize_filename()

    def initialize_filename(self):
        self.file_index = 0
        self.entry_index = -1
        self.filename = "{}.{}.npz".format(self.prefix, self.file_index)

    def update_filename(self):
        self.file_index += 1
        self.entry_index = 0
        self.filename = "{}.{}.npz".format(self.prefix, self.file_index)

    def get_next_entry(self):
        self.entry_index += 1
        if self.entry_index >= self.batch_size:
            self.update_filename()
        return {"path": self.filename, "index": self.entry_index}

    def get_batch_entries(self, num_samples, TF_id, CT_id):
        file_index = 0
        result = {}
        for start in range(0, num_samples, self.batch_size):
            stop = min(num_samples, start + self.batch_size)
            filename = "{}.{}.npz".format(self.prefix, file_index)
            result[filename] = {
                "start": start,
                "stop": stop,
                "TF_id": TF_id,
                "CT_id": CT_id,
            }
            file_index += 1
        return result


class FeatureWriter(object):
    def __init__(
        self,
        batch_size,
        seq_length,
        num_samples,
        group_name,
        feature_kwd,
        num_features=1,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.group_name = group_name
        self.feature_kwd = feature_kwd
        self.num_features = num_features
        # Initialize
        self.start_time = time.time()
        self.path = None
        self.paths = []
        self.counter = 0

    def write_feature(self, signal, current_index, entry, **kwargs):
        assert current_index == self.counter
        path, index = entry["path"], entry["index"]
        # Update path
        if index <= 0:
            self.path = path
            self.__initialize_array()

        assert self.path == path
        self.array[index] = signal
        self.counter += 1

        if index + 1 >= self.array.shape[0]:
            self.__write_to_disk(**kwargs)
            self.paths.append(path)

    def get_paths(self):
        return self.paths

    def __write_to_disk(self, **kwargs):
        assert not np.isnan(self.array).any()
        result = {
            "group_name": self.group_name,
            "feature_type": self.feature_kwd,
            "start": self.start,
            "stop": self.stop,
            "original": self.array,
        }
        if any(kwargs):
            assert len(kwargs) == 1
            result = {**result, **kwargs}
            key = next(iter(kwargs))
            self.path = self.path.replace(
                ".npz", ".{}{:.0e}.npz".format(key, kwargs[key])
            )
        np.savez_compressed(self.path, **result)
        print_time(
            "Samples {} to {} saved in {}".format(
                self.start, self.stop, self.path
            ),
            self.start_time,
        )

    def __initialize_array(self):
        batch_size = min(self.batch_size, self.num_samples - self.counter)
        self.start = self.counter
        self.stop = self.counter + batch_size
        self.array = np.empty((batch_size, self.seq_length, self.num_features))
        self.array = self.array.squeeze()
        self.array[:] = np.NaN


class SignalNormalizer(object):
    def __init__(self, method, save_params=None, mu=None, std=None):
        if method == "zscore":
            self.method = method
            self.mu = mu
            self.std = std
            if save_params is not None:
                np.savez_compressed(save_params, mu=self.mu, std=self.std)
                print("Zscore params saved in {}".format(save_params))
            self.normalize = self.zscore_signal

    def zscore_signal(self, signal):
        return (signal - self.mu) / self.std


class FeatureReader(object):
	def __init__(self, dset_name, threshold=None):
		self.dset_name = dset_name
		self.threshold = threshold
		self.path = None
		self.array = None

	def read_feature(self, metadata):
		if self.threshold is None:
			realpath = metadata['path']
		else:
			realpath = metadata['path'].replace('.npz',
				'.threshold{:.0e}.npz'.format(self.threshold))

		if self.path != realpath:
			self.path = realpath
			print("Read from a NEW file {}".format(self.path))
			self.array = np.load(self.path)[self.dset_name]

		return self.array[metadata['index']]


class BatchMeanCalculator(object):
    def __init__(self):
        self.running_sum = 0
        self.num_samples = 0
        
    def add_batch_data(self, batch):
        self.running_sum += batch.sum()
        self.num_samples += len(batch.flatten())

    def get_mean(self):
        return self.running_sum / self.num_samples    
    
    
class BatchVarianceCalculator(object):
    def __init__(self, mu):
        self.mu = mu
        self.num_samples = 0
        self.sum_of_square = 0
        
    def add_batch_data(self, batch):
        batch = batch.flatten()
        self.num_samples += len(batch)
        self.sum_of_square += ((batch - self.mu) ** 2).sum()
        
    def get_variance(self):
        return self.sum_of_square / self.num_samples
    
    def get_standard_deviation(self):
        return np.sqrt(self.get_variance())


def print_time(msg, start_time):
    elapse = time.strftime(
        "%H:%M:%S", time.gmtime(int((time.time() - start_time)))
    )
    print("%s. Time elapse %s" % (msg, elapse))
    sys.stdout.flush()
    return time.time()


def display_args(args, path):
    print("Running {}.".format(os.path.basename(path)))
    print(
        "CONFIG:\n{}".format(json.dumps(vars(args), indent=4, sort_keys=True))
    )

def get_bucket_id(sample_id):
    return str(int(sample_id // BUCKET_SIZE))