import os
import random
import numpy as np
import pandas as pd
import torch
from scipy import signal
from tqdm import tqdm


def min_max_normalize(arr):
    try:
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val)
    except:
        return arr

def z_score_normalize(arr):
    mu = np.nanmean(arr, axis=-1, keepdims=True)
    std = np.nanstd(arr, axis=-1, keepdims=True)
    return (arr - mu) / std

def find_challenge_files(data_directory_list):
    header_files = list()
    recording_files = list()
    for data_directory in data_directory_list:
        for f in sorted(os.listdir(data_directory)):
            root, extension = os.path.splitext(f)
            if not root.startswith('.') and extension=='.hea':
                header_file = os.path.join(data_directory, root + '.hea')
                recording_file = os.path.join(data_directory, root + '.mat')
                if os.path.isfile(header_file) and os.path.isfile(recording_file):
                    header_files.append(header_file)
                    recording_files.append(recording_file)
    return header_files, recording_files

# Adapted from original scoring function code
# For each set of equivalent classes, replace each class with the representative class for the set.
def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                classes[j] = multiple_classes[0]  # Use the first class as the representative class.
    return classes


# Load header file as a string.
def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key='val'):
    from scipy.io import loadmat
    recording = loadmat(recording_file)[key]
    if header and leads:
        recording = choose_leads(recording, header, leads)
    return recording

# Choose leads from the recording file.
def choose_leads(recording, header, leads):
    num_leads = len(leads)
    num_samples = np.shape(recording)[1]
    chosen_recording = np.zeros((num_leads, num_samples), recording.dtype)
    available_leads = get_leads(header)
    for i, lead in enumerate(leads):
        if lead in available_leads:
            j = available_leads.index(lead)
            chosen_recording[i, :] = recording[j, :]
    return chosen_recording

# Get recording ID.
def get_recording_id(header):
    recording_id = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                recording_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return recording_id

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get age from header.
def get_age(header):
    age = None
    for l in header.split('\n'):
        if l.startswith('#Age'):
            try:
                age = float(l.split(': ')[1].strip())
            except:
                age = float('nan')
    return age

# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('#Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get frequency from header.
def get_num_leads(header):
    num_leads = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_samples = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_leads

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get number of samples from header.
def get_num_samples(header):
    num_samples = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                num_samples = float(l.split(' ')[3])
            except:
                pass
        else:
            break
    return num_samples

# Get labels from header.
def get_labels(header, classes_to_skip):
    labels = list()
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    if entry not in classes_to_skip:
                        labels.append(entry.strip())
            except:
                pass
    return labels

dataset_labels = {
    "ptbxl": 0,
    "cpsc": 1,
    "cpsc_2": 1,
    "ga": 2,
    "chapman": 3,
    "ningbo":4
}

# Database folder names, matched against the parent directory of each header file
dataset_paths = {
    "ptbxl": 'WFDB_PTBXL',
    "cpsc": 'WFDB_CPSC2018',
    "cpsc_2": 'WFDB_CPSC2018_2',
    "ga": 'WFDB_Ga',
    "chapman": 'WFDB_ChapmanShaoxing',
    "ningbo": 'WFDB_Ningbo'
}
def get_class_source(file_path):
    folder = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    for dataset, paths in dataset_paths.items():
        if folder == paths:
            return dataset_labels[dataset]
    return None  # Return None if no match is found

def get_nsamp(header):
    return int(header.split('\n')[0].split(' ')[3])



class dataset:

    classes = ['164889003', #AF
               '426783006', '426177001', '427084000', #SR
               '57054005', '54329005', '164865005'] #MI

    diseases_names = {'164889003': 'atrial fibrillation',
                      '426783006': 'sinus rhythm',
                      '426177001': 'sinus bradycardia',
                      '427084000': 'sinus tachycardia',
                      '57054005': 'acute myocardial infarction',
                      '54329005': 'anterior myocardial infarction',
                      '164865005': 'myocardial infarction'}
    classes_to_skip = []
    normal_class = '426783006'


    def __init__(self, header_files, length, classes_to_skip=[], nr_leads=None, normalize=False, return_source=False, equivalent_cl='sinus_mi'):
        self.files = []
        self.sample = True
        self.num_leads = nr_leads
        self.classes_to_skip = classes_to_skip
        self.length = length
        self.normalize = normalize
        self.return_source = return_source
        self.equivalent_cl = equivalent_cl

        if self.equivalent_cl == 'sinus_mi':
            self.equivalent_classes = [['426783006', '426177001', '427084000'],
                                       ['164865005', '57054005', '54329005']]
        elif self.equivalent_cl == 'sinus':
            self.equivalent_classes = [['426783006', '426177001', '427084000']]
        elif self.equivalent_cl is None:
            self.equivalent_classes = []

        if self.classes_to_skip:
            dataset.classes = [c for c in dataset.classes if c not in self.classes_to_skip]
        for h in tqdm(header_files):
            tmp = dict()
            tmp['header'] = h
            tmp['record'] = h.replace('.hea', '.mat')
            hdr = load_header(h)
            tmp['nsamp'] = get_nsamp(hdr)
            tmp['leads'] = get_leads(hdr)
            tmp['age'] = get_age(hdr)
            tmp['sex'] = get_sex(hdr)
            tmp['dx'] = get_labels(hdr, dataset.classes_to_skip)
            tmp['fs'] = get_frequency(hdr)
            tmp['target'] = np.zeros((len(dataset.classes),))  # 26
            tmp['source'] = get_class_source(h)
            tmp['dx'] = replace_equivalent_classes(tmp['dx'], self.equivalent_classes)
            for dx in tmp['dx']:
                if dx in dataset.classes:
                    idx = dataset.classes.index(dx)
                    tmp['target'][idx] = 1
            self.files.append(tmp)
        self.files = pd.DataFrame(self.files)

    def summary(self, output):
        if output == 'pandas':
            summary_series = pd.Series(np.stack(self.files['target'].to_list(), axis=0).sum(axis=0),
                                       index=dataset.classes)
            summary_series.index = summary_series.index.map(self.diseases_names)
            return summary_series
        if output == 'numpy':
            return np.stack(self.files['target'].to_list(), axis=0).sum(axis=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fs = self.files.iloc[item]['fs']
        if self.equivalent_cl == 'sinus_mi':
            af_index = dataset.classes.index('164889003')
            sr_index = dataset.classes.index('426783006')
            mi_index = dataset.classes.index('164865005')
            target = self.files.iloc[item]['target'][[af_index, sr_index, mi_index]]
        elif self.equivalent_cl == 'sinus':
            af_index = dataset.classes.index('164889003')
            sr_index = dataset.classes.index('426783006')
            target = self.files.iloc[item]['target'][[af_index, sr_index]]
        else:
            target = self.files.iloc[item]['target']
        leads = self.files.iloc[item]['leads']
        sex = self.files.iloc[item]['sex']
        age = self.files.iloc[item]['age']
        data = load_recording(self.files.iloc[item]['record'])
        source = self.files.iloc[item]['source']

        # expand to 12 lead setup if original signal has less channels
        data, lead_indicator = expand_leads(data, input_leads=leads)
        data = np.nan_to_num(data)

        # resample to 500hz
        if fs == float(1000):
            data = signal.resample_poly(data, up=1, down=2, axis=-1)  # to 500Hz
            fs = 500
        elif fs == float(500):
            pass
        else:
            data = signal.resample(data, int(data.shape[1] * 500 / fs), axis=1)
            fs = 500

        if self.sample:
            fs = int(fs)
            if data.shape[-1] > self.length:
                idx = np.random.randint(data.shape[-1] - self.length + 1)
                data = data[:, idx:idx + self.length]
            if data.shape[-1] < self.length:
                def extend_array_with_zeros(array, target_length):
                    extended_array = np.zeros(target_length, dtype=array.dtype)
                    extended_array[:len(array)] = array
                    return extended_array

                data = np.array([extend_array_with_zeros(arr, self.length) for arr in data])

        # data normalization
        if self.normalize == 'min_max':
            data = np.array([min_max_normalize(arr) for arr in data])
        if self.normalize == 'z-score':
            data = np.array([z_score_normalize(arr) for arr in data])



        data = np.nan_to_num(data)

        # random choose number of leads to keep
        data, lead_indicator = lead_exctractor.get(data, self.num_leads, lead_indicator)
        if self.return_source:
            return data, target, lead_indicator, source
        else:
            return data, target, lead_indicator

    def get_item_with_sex_age(self, item):
        if self.return_source:
            data, target, lead_indicator, source = self.__getitem__(item)
        else:
           data, target, lead_indicator = self.__getitem__(item)
        sex = self.files.iloc[item]['sex']
        age = self.files.iloc[item]['age']
        return data, target, lead_indicator, sex, age

    def get_all_lables(self):
        targets = []
        for i in range(len(self)):
            _, target, _ = self[i]
            targets.append(target)
        return np.array(targets)


def expand_leads(recording,input_leads):
    output = np.zeros((12, recording.shape[1]))
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    twelve_leads = [k.lower() for k in twelve_leads]

    input_leads = [k.lower() for k in input_leads]
    output_leads = np.zeros((12,))
    for i,k in enumerate(input_leads):
        idx = twelve_leads.index(k)
        output[idx,:] = recording[i,:]
        output_leads[idx] = 1
    return output, output_leads


class lead_exctractor:
    """
    used to select specific leads or random choice of configurations

    Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Six leads: I, II, III, aVR, aVL, aVF
    Four leads: I, II, III, V2
    Three leads: I, II, V2
    Two leads: I, II

   """

    L1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    L2 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    L3 = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    L4 = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    L6 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    L8 = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1])
    L12 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    @staticmethod
    def get (x,num_leads,lead_indicator):
        if num_leads=='random_choice':
            # random choice output
            num_leads = random.choice([12,6,4,3,2])

        if num_leads=='custom_amc':
            # random choice output
            num_leads = random.choice([12, 8, 2,'prob_selection'])

        if num_leads == 'prob_selection':
            # Select each lead with a probability of 0.5
            selected_leads = np.random.choice([0, 1], size=12, p=[0.5, 0.5])
            x = x * selected_leads.reshape(12, 1)
            return x, lead_indicator * selected_leads

        if num_leads==12:
            # Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
            return x,lead_indicator * lead_exctractor.L12

        if num_leads==6:
            # Six leads: I, II, III, aVL, aVR, aVF
            x = x * lead_exctractor.L6.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L6

        if num_leads==8:
            # Eight leads: II, aVR, V1-V6
            x = x * lead_exctractor.L8.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L8

        if num_leads==4:
            # Six leads: I, II, III, V2
            x = x * lead_exctractor.L4.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L4

        if num_leads==3:
            # Three leads: I, II, V2
            x = x * lead_exctractor.L3.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L3

        if num_leads==2:
            x = x * lead_exctractor.L2.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L2

        if num_leads==1:
            x = x * lead_exctractor.L1.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L1

        raise Exception("invalid-leads-number")


def classes_info(dataset):
    summary_series = dataset.summary('pandas')
    if dataset.equivalent_classes == [['426783006', '426177001', '427084000']]:
        filtered_summary = summary_series[['atrial fibrillation', 'sinus rhythm']]
        print('Total number of samples:', len(dataset))
        print(filtered_summary)
        af_index = dataset.classes.index('164889003')
        sr_index = dataset.classes.index('426783006')
        print('AF index:', af_index)
        print('SR index:', sr_index)
    else:
        filtered_summary = summary_series[['atrial fibrillation', 'sinus rhythm', 'myocardial infarction']]
        print('Total number of samples:', len(dataset))
        print(filtered_summary)
        af_index = dataset.classes.index('164889003')
        sr_index = dataset.classes.index('426783006')
        mi_index = dataset.classes.index('164865005')

        print('AF index:', af_index)
        print('SR index:', sr_index)
        print('Myocardial infarction index:', mi_index)


def collate(batch):
    ch = batch[0][0].shape[0]
    maxL = batch[0][0].shape[1]
    X = np.zeros((len(batch), ch, maxL))
    for i in range(len(batch)):
        X[i, :, -batch[i][0].shape[-1]:] = batch[i][0]
    t = np.array([b[1] for b in batch])
    l = np.concatenate([b[2].reshape(1,12) for b in batch],axis=0)

    X = torch.from_numpy(X)
    t = torch.from_numpy(t)
    l = torch.from_numpy(l)
    return X, t, l

class dataset_extended_diagnostic:
    classes = [
        '164889003', '164890007', '733534002', '426627000', '6374002',
        '713427006', '270492004', '713426002', '39732003', '445118002',
        '164947007', '251146004', '111975006', '698252002', '426783006',
        '284470004', '10370003', '427172004', '164917005', '365413008',
        '47665007', '427393009', '426177001', '427084000', '164934002',
        '59931005'
    ]

    disease_names = {
        '164889003': 'Atrial Fibrillation', '164890007': 'Atrial Flutter',
        '733534002': 'Complete Left Bundle Branch Block', '426627000': 'Bradycardia',
        '6374002': 'Bundle Branch Block', '713427006': 'Complete Right Bundle Branch Block',
        '270492004': '1st Degree AV Block', '713426002': 'Incomplete Right Bundle Branch Block',
        '39732003': 'Left Axis Deviation', '445118002': 'Left Anterior Fascicular Block',
        '164947007': 'Prolonged PR Interval', '251146004': 'Low QRS Voltages',
        '111975006': 'Prolonged QT Interval', '698252002': 'Nonspecific Intraventricular Conduction Disorder',
        '426783006': 'Sinus Rhythm', '284470004': 'Premature Atrial Contraction',
        '10370003': 'Pacing Rhythm', '365413008': 'Poor R Wave Progression',
        '427172004': 'Premature Ventricular Contractions', '164917005': 'Q Wave Abnormality',
        '47665007': 'Right Axis Deviation', '427393009': 'Sinus Arrhythmia',
        '426177001': 'Sinus Bradycardia', '427084000': 'Sinus Tachycardia',
        '164934002': 'T Wave Abnormality', '59931005': 'T Wave Inversion'
    }

    key_conditions = {
        '164889003': 'Atrial Fibrillation',
        '426783006': 'Sinus Rhythm',
        '427084000': 'Sinus Tachycardia',
        '426177001': 'Sinus Bradycardia'
    }

    equivalent_classes = [
        ['713427006', '59118001'], ['284470004', '63593006'],
        ['427172004', '17338001'], ['733534002', '164909002']
    ]

    def __init__(self, header_files, classes_to_skip=[]):
        self.classes_to_skip = set(classes_to_skip)
        self.files = self._process_headers(header_files)

    def _process_headers(self, header_files):
        files = []
        for h in tqdm(header_files, desc="Processing headers"):
            hdr = load_header(h)
            dx = get_labels(hdr, self.classes_to_skip)
            dx = self._replace_equivalent_classes(dx)

            target = np.zeros(len(dataset_extended_diagnostic.classes))
            for d in dx:
                if d in dataset_extended_diagnostic.classes:
                    target[dataset_extended_diagnostic.classes.index(d)] = 1

            diseases = [dataset_extended_diagnostic.disease_names[d] for d in dx if d in dataset_extended_diagnostic.disease_names]

            files.append({'header': h, 'diseases': diseases, 'target': target})

        return pd.DataFrame(files)

    def _replace_equivalent_classes(self, dx):
        for group in dataset_extended_diagnostic.equivalent_classes:
            if any(c in dx for c in group):
                dx = [group[0] if c in group else c for c in dx]
        return dx

    def summary(self):
        disease_counts = np.stack(self.files['target']).sum(axis=0)
        summary_dict = {dataset_extended_diagnostic.disease_names[cls]: int(disease_counts[i])
                        for i, cls in enumerate(dataset_extended_diagnostic.classes) if disease_counts[i] > 0}

        # Co-occurrence tracking
        co_occurrences = {name: {} for name in dataset_extended_diagnostic.key_conditions.values()}

        for _, row in self.files.iterrows():
            present_diseases = row['diseases']
            present_keys = [name for code, name in dataset_extended_diagnostic.key_conditions.items() if name in present_diseases]

            for key in present_keys:
                for disease in present_diseases:
                    if disease != key:
                        co_occurrences[key][disease] = co_occurrences[key].get(disease, 0) + 1

        print("\n=== Disease Counts ===")
        for disease, count in summary_dict.items():
            print(f"{disease}: {count}")

        print("\n=== Co-occurrence with Key Conditions ===")
        for key_condition, co_list in co_occurrences.items():
            print(f"\n- {key_condition}:")
            for disease, count in sorted(co_list.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {disease}: {count} times")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.files.iloc[idx]
        return row['header'], row['diseases']